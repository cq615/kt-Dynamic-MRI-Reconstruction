import torch
import torch.nn as nn
from torch.autograd import Variable


def _fftshift(x, axes, offset=1):
    """ Apply ifftshift to x.

    Parameters:
    -----------

    x: torch.Tensor

    axes: tuple. axes to apply ifftshift. E.g.: axes=(-1), axes=(2,3), etc..

    Returns:
    --------

    result of applying ifftshift(x, axes).

    """
    # build slice
    x_shape = x.shape
    ndim = len(x_shape)
    axes = [ (ndim + ax) % ndim for ax in axes ]

    # apply shift for each axes:
    for ax in axes:
        # build slice:
        if x_shape[ax] == 1:
            continue
        n = x_shape[ax]
        half_n = (n + offset)//2
        curr_slice = [ slice(0, half_n) if i == ax else slice(x_shape[i]) for i in range(ndim) ]
        curr_slice_2 = [ slice(half_n, x_shape[i]) if i == ax else slice(x_shape[i]) for i in range(ndim) ]
        x = torch.cat([x[curr_slice_2], x[curr_slice]], dim=ax)
    return x


def fftshift_pytorch(x, axes):
    return _fftshift(x, axes, offset=1)


def ifftshift_pytorch(x, axes):
    return _fftshift(x, axes, offset=0)


def data_consistency(k, k0, mask, noise_lvl=None):
    """
    k    - input in k-space
    k0   - initially sampled elements in k-space
    mask - corresponding nonzero location
    """
    v = noise_lvl
    if v:  # noisy case
        out = (1 - mask) * k + mask * (k + v * k0) / (1 + v)
    else:  # noiseless case
        out = (1 - mask) * k + mask * k0
    return out


class DataConsistencyInKspace(nn.Module):
    """ Create data consistency operator

    Warning: note that FFT2 (by the default of torch.fft) is applied to the last 2 axes of the input.
    This method detects if the input tensor is 4-dim (2D data) or 5-dim (3D data)
    and applies FFT2 to the (nx, ny) axis.

    """

    def __init__(self, noise_lvl=None, norm='ortho'):
        super(DataConsistencyInKspace, self).__init__()
        self.normalized = norm == 'ortho'
        self.noise_lvl = noise_lvl

    def forward(self, *input, **kwargs):
        return self.perform(*input)

    def perform(self, x, k0, mask):
        """
        x    - input in image domain, of shape (n, 2, nx, ny[, nt])
        k0   - initially sampled elements in k-space
        mask - corresponding nonzero location
        """

        if x.dim() == 4: # input is 2D
            x    = x.permute(0, 2, 3, 1)
            k0   = k0.permute(0, 2, 3, 1)
            mask = mask.permute(0, 2, 3, 1)
        elif x.dim() == 5: # input is 3D
            x    = x.permute(0, 4, 2, 3, 1)
            k0   = k0.permute(0, 4, 2, 3, 1)
            mask = mask.permute(0, 4, 2, 3, 1)

        k = torch.fft(x, 2, normalized=self.normalized)
        out = data_consistency(k, k0, mask, self.noise_lvl)
        x_res = torch.ifft(out, 2, normalized=self.normalized)

        if x.dim() == 4:
            x_res = x_res.permute(0, 3, 1, 2)
        elif x.dim() == 5:
            x_res = x_res.permute(0, 4, 2, 3, 1)

        return x_res


class CRNNcell(nn.Module):
    """
    Convolutional RNN cell that evolves over both time and iterations

    Parameters
    -----------------
    input: 4d tensor, shape (batch_size, channel, width, height)
    hidden: hidden states in temporal dimension, 4d tensor, shape (batch_size, hidden_size, width, height)
    hidden_iteration: hidden states in iteration dimension, 4d tensor, shape (batch_size, hidden_size, width, height)
    iteration: True or False, to use iteration recurrence or not; if iteration=False: hidden_iteration=None

    Returns
    -----------------
    output: 4d tensor, shape (batch_size, hidden_size, width, height)

    """
    def __init__(self, input_size, hidden_size, kernel_size, dilation, iteration=False):
        super(CRNNcell, self).__init__()
        self.kernel_size = kernel_size
        self.iteration = iteration
        self.i2h = nn.Conv2d(input_size, hidden_size, kernel_size, padding=dilation, dilation=dilation)
        self.h2h = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=dilation, dilation=dilation)
        # add iteration hidden connection
        if self.iteration:
            self.ih2ih = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        self.relu = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, input, hidden, hidden_iteration=None):
        in_to_hid = self.i2h(input)
        hid_to_hid = self.h2h(hidden)
        if hidden_iteration is not None:
            ih_to_ih = self.ih2ih(hidden_iteration)
            hidden = self.relu(in_to_hid + hid_to_hid + ih_to_ih)
        else:
            hidden = self.relu(in_to_hid + hid_to_hid)

        return hidden


class BCRNNlayer(nn.Module):
    """
    Bidirectional Convolutional RNN layer

    Parameters
    --------------------
    incomings: input: 5d tensor, [input_image] with shape (num_seqs, batch_size, channel, width, height)
               input_iteration: 5d tensor, [hidden states from previous iteration] with shape (n_seq, n_batch, hidden_size, width, height)
               test: True if in test mode, False if in train mode
               iteration: True if use iteration recurrence and input_iteration is not None; False if input_iteration=None

    Returns
    --------------------
    output: 5d tensor, shape (n_seq, n_batch, hidden_size, width, height)

    """
    def __init__(self, input_size, hidden_size, kernel_size, dilation, iteration=False):
        super(BCRNNlayer, self).__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.iteration = iteration
        self.CRNN_model = CRNNcell(self.input_size, self.hidden_size, self.kernel_size, dilation, iteration=self.iteration)

    def forward(self, input, input_iteration=None, test=False):
        nt, nb, nc, nx, ny = input.shape
        size_h = [nb, self.hidden_size, nx, ny]
        if test:
            with torch.no_grad():
                hid_init = Variable(torch.zeros(size_h), requires_grad=False).cuda()
        else:
            hid_init = Variable(torch.zeros(size_h), requires_grad=False).cuda()

        output_f = []
        output_b = []
        if input_iteration is not None:
            # forward
            hidden = hid_init
            for i in range(nt):
                hidden = self.CRNN_model(input[i], hidden, input_iteration[i])
                output_f.append(hidden)
            # backward
            hidden = hid_init
            for i in range(nt):
                hidden = self.CRNN_model(input[nt - i - 1], hidden, input_iteration[nt - i -1])
                output_b.append(hidden)
        else:
            # forward
            hidden = hid_init
            for i in range(nt):
                hidden = self.CRNN_model(input[i], hidden)
                output_f.append(hidden)
            # backward
            hidden = hid_init
            for i in range(nt):
                hidden = self.CRNN_model(input[nt - i - 1], hidden)
                output_b.append(hidden)

        output_f = torch.cat(output_f)
        output_b = torch.cat(output_b[::-1])

        output = output_f + output_b

        if nb == 1:
            output = output.view(nt, 1, self.hidden_size, nx, ny)

        return output


class TransformDataInXfSpaceTA(nn.Module):

    def __init__(self, divide_by_n=False, norm=True):
        super(TransformDataInXfSpaceTA, self).__init__()
        self.normalized = norm
        self.divide_by_n = divide_by_n

    def forward(self, x, k0, mask):
        return self.perform(x, k0, mask)

    def perform(self, x, k0, mask):
        """
        transform to x-f space with subtraction of average temporal frame
        :param x: input image with shape [n, 2, nx, ny, nt]
        :param mask: undersampling mask
        :return: difference data; DC baseline
        """
        # temporally average kspace and image data
        x = x.permute(0, 4, 2, 3, 1)
        mask = mask.permute(0, 4, 2, 3, 1)
        k0 = k0.permute(0, 4, 2, 3, 1)
        k = torch.fft(x, 2, normalized=self.normalized)
        if self.divide_by_n:
            k_avg = torch.div(torch.sum(k, 1), k.shape[1])
        else:
            k_avg = torch.div(torch.sum(k, 1), torch.clamp(torch.sum(mask, 1), min=1))

        nb, nx, ny, nc = k_avg.shape
        k_avg = k_avg.view(nb, 1, nx, ny, nc)
        # repeat the temporal frame and
        k_avg = k_avg.repeat(1, k.shape[1], 1, 1, 1)

        # subtract the temporal average frame
        k_diff = torch.sub(k, k_avg)
        x_diff = torch.ifft(k_diff, 2, normalized=self.normalized)

        # transform to x-f space to get the baseline
        k_avg = data_consistency(k_avg, k0, mask)
        x_avg = torch.ifft(k_avg, 2, normalized=self.normalized)

        x_avg = x_avg.permute(0, 2, 3, 1, 4)  # [n, nx, ny, nt, 2]
        x_f_avg = fftshift_pytorch(torch.fft(ifftshift_pytorch(x_avg, axes=[-2]), 1, normalized=self.normalized), axes=[-2])
        x_f_avg = x_f_avg.permute(0, 4, 1, 2, 3)

        # difference data
        x_diff = x_diff.permute(0, 2, 3, 1, 4)  # [n, nx, ny, nt, 2]
        x_f_diff = fftshift_pytorch(torch.fft(ifftshift_pytorch(x_diff, axes=[-2]), 1, normalized=self.normalized), axes=[-2])
        x_f_diff = x_f_diff.permute(0, 4, 1, 2, 3)

        return x_f_diff, x_f_avg

