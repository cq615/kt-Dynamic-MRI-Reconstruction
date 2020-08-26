import torch
import numpy as np
import argparse, os

from torch.autograd import Variable
import torch.optim as optim
from scipy.io import loadmat

import time
import gc
from utils import compressed_sensing as cs
from utils.metric import complex_psnr

from network.kt_NEXT import *
from utils.dnn_io import to_tensor_format
from utils.dnn_io import from_tensor_format
from numpy.fft import fft, fft2, ifftshift, fftshift
import matplotlib.pyplot as plt

def prep_input(im, acc=4):
    """Undersample the batch, then reformat them into what the network accepts.
    Parameters
    ----------
    gauss_ivar: float - controls the undersampling rate.
                        higher the value, more undersampling
    """
    mask = cs.shear_grid_mask(im.shape[1:], acc, sample_low_freq=True, sample_n=4)
    mask = np.repeat(mask[np.newaxis], im.shape[0], axis=0)

    im_und, k_und = cs.undersample(im, mask, centred=False, norm='ortho')
    im_gnd_l = torch.from_numpy(to_tensor_format(im))
    im_und_l = torch.from_numpy(to_tensor_format(im_und))
    k_und_l = torch.from_numpy(to_tensor_format(k_und))
    mask_l = torch.from_numpy(to_tensor_format(mask, mask=True))

    im = im.transpose(0, 2, 3, 1)
    xf_gnd = fftshift(fft(ifftshift(im, axes=-1), norm='ortho'), axes=-1)
    xf_gnd = xf_gnd.transpose(0, 3, 1, 2)
    xf_gnd_l = torch.from_numpy(to_tensor_format(xf_gnd))

    return im_und_l, k_und_l, mask_l, im_gnd_l, xf_gnd_l


def iterate_minibatch(data, batch_size, shuffle=True):
    n = len(data)

    if shuffle:
        data = np.random.permutation(data)

    for i in range(0, n, batch_size):
        yield data[i:i+batch_size]


def create_dummy_data():
    """Create small cardiac data based on patches for demo.
    Note that in practice, at test time the method will need to be applied to
    the whole volume. In addition, one would need more data to prevent
    overfitting.
    """
    data = loadmat(os.path.join(project_root, './data/cardiac.mat'))['seq']
    nx, ny, nt = data.shape
    ny_red = 8
    sl = ny//ny_red
    data_t = np.transpose(data, (2, 0, 1))
    
    # Synthesize data by extracting patches
    train = np.array([data_t[..., i:i+sl] for i in np.random.randint(0, sl*3, 20)])
    validate = np.array([data_t[..., i:i+sl] for i in (sl*4, sl*5)])
    test = np.array([data_t[..., i:i+sl] for i in (sl*6, sl*7)])

    return train, validate, test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epoch', metavar='int', nargs=1, default=['10'],
                        help='number of epochs')
    parser.add_argument('--batch_size', metavar='int', nargs=1, default=['1'],
                        help='batch size')
    parser.add_argument('--lr', metavar='float', nargs=1,
                        default=['0.001'], help='initial learning rate')
    parser.add_argument('--acceleration_factor', metavar='int', nargs=1,
                        default=['4'],
                        help='Acceleration factor for k-space sampling')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    
    args = parser.parse_args()

    # Project config
    model_name = 'kt_NEXT'
    acc = int(args.acceleration_factor[0])  # undersampling rate
    num_epoch = int(args.num_epoch[0])
    batch_size = int(args.batch_size[0])
    Nx, Ny, Nt = 256, 256, 30
    Ny_red = 8
    save_every = 5

    # Configure directory info
    project_root = '.'
    save_dir = os.path.join(project_root, 'models')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Create dataset
    train, validate, test = create_dummy_data()

    # cuda = True if torch.cuda.is_available() else False
    cuda = True
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    nc = 4
    # build the model
    xf_net = kt_NEXT_model(nc=nc)
    criterion = torch.nn.MSELoss()
    if cuda:
        xf_net = xf_net.cuda()
        criterion.cuda()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, xf_net.parameters()), lr=float(args.lr[0]), betas=(0.5, 0.999))
    pytorch_total_params = sum(p.numel() for p in xf_net.parameters() if p.requires_grad)
    print('Total trainable params: %d' % pytorch_total_params)

    for epoch in range(0, num_epoch+1):
        gc.collect()
        t_start = time.time()
        train_err = 0
        train_batches = 0

        for im in iterate_minibatch(train, batch_size, shuffle=True):
            x_und, k_und, mask, x_gnd, xf_gnd = prep_input(im, acc)
            x_u = Variable(x_und.type(Tensor))
            k_u = Variable(k_und.type(Tensor))
            mask = Variable(mask.type(Tensor))
            gnd = Variable(x_gnd.type(Tensor))
            xf_gnd = Variable(xf_gnd.type(Tensor))

            optimizer.zero_grad()
            xf_out, img = xf_net(x_u, k_u, mask)

            loss = criterion(img['t%d' % (nc - 1)], gnd) + criterion(xf_out['t%d' % (nc-1)], xf_gnd)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(xf_net.parameters(), 5)
            optimizer.step()

            train_err += loss.item()
            train_batches += 1

        t_end = time.time()
        train_err /= train_batches

        if epoch % 1 == 0:

            xf_net.eval()
            test_loss = []
            base_psnr = []
            epoch_psnr = []

            for im in iterate_minibatch(validate, batch_size, shuffle=False):
                x_und, k_und, mask, x_gnd, xf_gnd = prep_input(im, acc)
                x_u = Variable(x_und.type(Tensor))
                k_u = Variable(k_und.type(Tensor))
                mask = Variable(mask.type(Tensor))
                gnd = Variable(x_gnd.type(Tensor))

                with torch.no_grad():
                    xf_out, img = xf_net(x_u, k_u, mask)

                test_loss.append(criterion(img['t%d' % (nc-1)], gnd).item())

                im_und = from_tensor_format(x_und.numpy())
                im_gnd = from_tensor_format(x_gnd.numpy())
                im_rec = from_tensor_format(img['t%d' % (nc-1)].data.cpu().numpy())

                for idx in range(im_und.shape[0]):
                    base_psnr.append(complex_psnr(im_gnd[idx], im_und[idx]))
                    epoch_psnr.append(complex_psnr(im_gnd[idx], im_rec[idx]))

            print("Epoch {}/{}".format(epoch + 1, num_epoch))
            print(" time: {}s".format(t_end - t_start))
            print(" training loss:\t\t{:.6f}".format(train_err))
            print(" testing loss:\t\t{:.6f}".format(np.mean(test_loss)))
            print(" base PSNR:\t\t{:.6f}".format(np.mean(base_psnr)))
            print(" test PSNR:\t\t{:.6f}".format(np.mean(epoch_psnr)))

            name = 'model_epoch_%d.npz' % epoch
            torch.save(xf_net.state_dict(), os.path.join(save_dir, name))
            print('model parameters saved at %s' % os.path.join(save_dir, name))
            print('')
