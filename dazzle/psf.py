from webbpsf import roman
from astropy.io import fits


def construct_psf(filter: str = 'F146', detector: str = 'SCA01',
                  position: tuple = (2048, 2048)) -> fits.hdu.hdulist.HDUList:
    """Construct a PSF for the indicated filter, detector, position, and oversampling factor."""

    wfi = roman.WFI()

    wfi.filter = filter
    wfi.detector = detector
    wfi.detector_position = position

    # Centre the PSF between pixels.
    wfi.options['parity'] = 'even'

    psf = wfi.calc_psf(oversample=10, normalize='last')

    return psf



if __name__ == '__main__':

    psf = construct_psf()
    print(psf)
    print(type(psf))
    psf.info()
    psf.writeto('test_psf.fits', overwrite=True)
