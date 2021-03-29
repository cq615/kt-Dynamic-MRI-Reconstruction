import torch
import numpy as np
import argparse, os
from torch.utils.data import DataLoader
import torch.utils.data as data

from torch.autograd import Variable
import torch.optim as optim

import time
import h5py

from utils.metric import complex_psnr
from network.CTFNet import *
from utils.dnn_io import *
from numpy.fft import fft2, ifft2


class data_loader(data.Dataset):
    """
    Demo data loader: pre-processed data file saved in .h5 format with keys: reference, coil_image and smaps and mask
    smaps: (n_subjects, n_coil, width, height)
    coil_img: (n_subjects, n_frame, n_coil, width, height)
    reference: (n_subjects, n_frame, width, height)
    mask: (n_subjects, n_frame, width, height)
    """
    def __init__(self, acc, transform=None):
        super(data_loader, self).__init__()
        self.transform=transform
        self.acc = acc

        # open directory
        self.data_base = './data/cardiac_mc.h5'  # save your multi_coil data in h5 file with the following keys
        self.data_smaps = h5py.File(self.data_base, 'r')['smaps'][:] # load sensitivity maps
        self.data_coil_img = h5py.File(self.data_base, 'r')['coil_img'][:] # load each coil image
        self.data_ref = h5py.File(self.data_base, 'r')['reference'][:] # load sensitivity weighted combined reference image
        self.data_masks = h5py.File(self.data_base, 'r')['mask'][:] # load undersampling mask
        self.n_subj = len(self.data_smaps)

        print("Dataset: {} elements".format(len(self)))

    def __getitem__(self, index):
        sample = {}
        sample['mask'] = self.data_masks[index]
        sample['smaps'] = self.data_smaps[index]
        sample['coil_img'] = self.data_coil_img[index]
        sample['ref'] = self.data_ref[index]

        return self.transform(sample)

    def __len__(self):
        return self.n_subj

def transform(args, test=False):
    """
    transforms and undersamples image
    """
    patch_ny = int(args.patch_size[0])    
    def transform(sample):
        coil_img = sample['coil_img']
        n_t, n_s = coil_img.shape[0], coil_img.shape[1]

        mask = sample['mask'][:].astype(np.int16)
        mask = np.tile(mask[:, np.newaxis], (1, n_s, 1, 1))

        smaps = sample['smaps']
        ref = sample['ref']

        if test is False:
            coil_img = coil_img.reshape(-1, coil_img.shape[2], coil_img.shape[3])
            # concatenate each coil image, sensitivity maps and reference for extracting patches
            comb = np.concatenate((coil_img, smaps, ref), axis=0)

            # extract patch in Ny direction
            max_ny = comb.shape[-2] - patch_ny + 1
            start_idx = np.random.randint(0, max_ny)
            start, end = start_idx, start_idx + patch_ny
            comb = comb[..., start:end, :]
            mask = mask[..., start:end, :]

            # recover the coil image, sensitivity maps and target
            coil_img = comb[:n_t * n_s]
            coil_img = coil_img.reshape(n_t, n_s, coil_img.shape[1], coil_img.shape[2])
            smaps = comb[n_t * n_s:-n_t]
            ref = comb[-n_t:]

        smaps = np.tile(smaps[np.newaxis], (n_t, 1, 1, 1))
        # generate undersampled data
        k_und = fft2(coil_img, axes=(-2, -1), norm='ortho') * mask
        x_und = np.sum(ifft2(k_und, axes=(-2, -1), norm='ortho') * np.conj(smaps), axis=(1))

        # HxWxT -> 2xHxWxT
        x_und = np.array([np.real(x_und), np.imag(x_und)], dtype=np.float32)
        k_und = np.array([np.real(k_und), np.imag(k_und)], dtype=np.float32)
        mask = np.array([mask, mask], dtype=np.float32)
        x_gnd = np.array([np.real(ref), np.imag(ref)], dtype=np.float32)
        x_smaps = np.array([np.real(smaps), np.imag(smaps)], dtype=np.float32)

        return x_und, k_und, mask, x_gnd, x_smaps

    return transform


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epoch', metavar='int', nargs=1, default=['10'],
                        help='number of epochs')
    parser.add_argument('--batch_size', metavar='int', nargs=1, default=['1'],
                        help='batch size')
    parser.add_argument('--patch_size', metavar='int', nargs=1, default=['64'],
                        help='patch size')
    parser.add_argument('--n_worker', metavar='int', nargs=1, default=['4'],
                        help='number of workers')
    parser.add_argument('--cascade', metavar='int', nargs=1, default=['2'],
                        help='number of cascades')
    parser.add_argument('--lr', metavar='float', nargs=1,
                        default=['0.0001'], help='initial learning rate')
    parser.add_argument('--acceleration_factor', metavar='int', nargs=1,
                        default=['8'],
                        help='Acceleration factor for k-space sampling')
    
    args = parser.parse_args()

    # Project config
    model_name = 'CTFNet'
    acc = int(args.acceleration_factor[0])  # undersampling rate
    n_epoch = int(args.num_epoch[0])
    n_worker = int(args.n_worker[0])
    bs = int(args.batch_size[0])
    lr=float(args.lr[0])
    cascade = int(args.cascade[0]) #stage number

    device = 'cuda:0'

    # Configure directory info
    project_root = '.'
    save_dir = os.path.join(project_root, 'models')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    dccoeff = 0.1 # data consistency layer parameter 
    beta = 0.1 # weighted coupling layer parameter 
    gamma = 0.1 # weighted coupling layer parameter 
    
    # build the model
    model = CTFNet_model(dccoeff, beta, gamma, cascade, bs).to(device)
    criterion = nn.L1Loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)

    train_set = data_loader(acc, transform=transform(args))
    test_set = data_loader(acc, transform=transform(args, test=True))

    training_data_loader = DataLoader(dataset=train_set, num_workers=n_worker,
                                      batch_size=bs, shuffle=True)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=n_worker,
                                     batch_size=bs, shuffle=False)

    for epoch in range(n_epoch+1):
        model.train()
        t_start = time.time()
        train_err = 0
        train_batches = 0

        for iteration, batch in enumerate(training_data_loader):

            x_und, k_und, masks, x_gnd, x_smaps = batch
            x_und = x_und.to(device)
            k_und = k_und.to(device)
            masks = masks.to(device)
            x_gnd = x_gnd.to(device)
            x_smaps = x_smaps.to(device)

            x_und = x_und.permute(2, 0, 3, 4, 1)     # n_seq, bs, width, height, n_ch
            k_und = k_und.permute(2, 0, 3, 4, 5, 1)  # n_seq, bs, n_coil, width, height, n_ch
            masks = masks.permute(2, 0, 3, 4, 5, 1)
            x_gnd = x_gnd.permute(2, 0, 3, 4, 1)
            x_smaps = x_smaps.permute(2, 0, 3, 4, 5, 1)

            rec = model(x_und, k_und, masks, x_smaps)

            loss = criterion(rec+1e-11, x_gnd) 

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            train_err += loss.item()
            train_batches += 1
            torch.cuda.empty_cache()
        
        t_end = time.time()
        train_err /= train_batches

        if epoch % 5 == 0:
            model.eval()
            test_loss = []
            base_psnr = []
            test_psnr = []

            for iteration, batch in enumerate(testing_data_loader):
                x_und, k_und, masks, x_gnd, x_smaps = batch

                x_und = x_und.to(device)
                k_und = k_und.to(device)
                masks = masks.to(device)
                x_gnd = x_gnd.to(device)
                x_smaps = x_smaps.to(device)

                x_und = x_und.permute(2, 0, 3, 4, 1)     # n_seq, bs, width, height, n_ch
                k_und = k_und.permute(2, 0, 3, 4, 5, 1)  # n_seq, bs, n_coil, width, height, n_ch
                masks = masks.permute(2, 0, 3, 4, 5, 1)
                x_gnd = x_gnd.permute(2, 0, 3, 4, 1)
                x_smaps = x_smaps.permute(2, 0, 3, 4, 5, 1)

                with torch.no_grad():
                    rec = model(x_und, k_und, masks, x_smaps, test=True)
                
                test_loss.append(criterion(rec+1e-11, x_gnd).item())

                sense_recon = r2c(rec.data.to('cpu').numpy(), axis=-1)
                sense_gt = r2c(x_gnd.data.to('cpu').numpy(), axis=-1)
                sense_und = r2c(x_und.data.to('cpu').numpy(), axis=-1)

                for idx in range(x_gnd.shape[1]):
                    base_psnr.append(complex_psnr(sense_gt[idx], sense_und[idx]))
                    test_psnr.append(complex_psnr(sense_gt[idx], sense_recon[idx]))

            print("Epoch {}/{}".format(epoch + 1, n_epoch))
            print(" time: {}s".format(t_end - t_start))
            print(" training loss:\t\t{:.6f}".format(train_err))
            print(" testing loss:\t\t{:.6f}".format(np.mean(test_loss)))
            print(" base PSNR:\t\t{:.6f}".format(np.mean(base_psnr)))
            print(" test PSNR:\t\t{:.6f}".format(np.mean(test_psnr)))

            name = 'CTFNet_epoch_%d.npz' % epoch
            torch.save(model.state_dict(), os.path.join(save_dir, name))
            print('model parameters saved at %s' % os.path.join(save_dir, name))
            print('')

