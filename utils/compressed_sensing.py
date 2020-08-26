import numpy as np
from numpy.fft import fft2, ifft2, ifftshift, fftshift


def shear_grid_mask(shape, acceleration_rate, sample_low_freq=False,
                    centred=False, sample_n=4, test=False):
    '''
    Creates undersampling mask which samples in sheer grid

    Parameters
    ----------

    shape: (nt, nx, ny)

    acceleration_rate: int

    Returns
    -------

    array

    '''
    Nt, Nx, Ny = shape
    if test:
        start = 0
    else:
        start = np.random.randint(0, acceleration_rate)
    mask = np.zeros((Nt, Nx))
    for t in range(Nt):
        mask[t, (start+t)%acceleration_rate::acceleration_rate] = 1

    xc = Nx // 2
    xl = sample_n // 2
    if sample_low_freq and centred:
        xh = xl
        if sample_n % 2 == 0:
            xh += 1
        mask[:, xc - xl:xc + xh+1] = 1

    elif sample_low_freq:
        xh = xl
        if sample_n % 2 == 1:
            xh -= 1

        if xl > 0:
            mask[:, :xl] = 1
        if xh > 0:
            mask[:, -xh:] = 1

    mask_rep = np.repeat(mask[..., np.newaxis], Ny, axis=-1)
    return mask_rep


def undersample(x, mask, centred=False, norm='ortho', noise=0):
    '''
    Undersample x. FFT2 will be applied to the last 2 axis
    Parameters
    ----------
    x: array_like
        data
    mask: array_like
        undersampling mask in fourier domain

    norm: 'ortho' or None
        if 'ortho', performs unitary transform, otherwise normal dft

    noise_power: float
        simulates acquisition noise, complex AWG noise.
        must be percentage of the peak signal

    Returns
    -------
    xu: array_like
        undersampled image in image domain. Note that it is complex valued

    x_fu: array_like
        undersampled data in k-space

    '''
    assert x.shape == mask.shape
    # zero mean complex Gaussian noise
    noise_power = noise
    nz = np.sqrt(.5)*(np.random.normal(0, 1, x.shape) + 1j * np.random.normal(0, 1, x.shape))
    nz = nz * np.sqrt(noise_power)

    if norm == 'ortho':
        # multiplicative factor
        nz = nz * np.sqrt(np.prod(mask.shape[-2:]))
    else:
        nz = nz * np.prod(mask.shape[-2:])

    if centred:
        axes = (-2, -1)
        x_f = fftshift(fft2(ifftshift(x, axes=axes), norm=norm), axes=axes)
        x_fu = mask * (x_f + nz)
        x_u = fftshift(ifft2(ifftshift(x_fu, axes=axes), norm=norm), axes=axes)
        return x_u, x_fu
    else:
        x_f = fft2(x, norm=norm)
        x_fu = mask * (x_f + nz)
        x_u = ifft2(x_fu, norm=norm)
        return x_u, x_fu

