import os
from webbpsf import roman
from webbpsf.utils import to_griddedpsfmodel
from astropy.io import fits
from photutils.psf import GriddedPSFModel


def construct_psf_grid(filter: str = 'F146', detector: str = 'SCA01', save: bool = False, outdir: str = None) -> GriddedPSFModel:
    """Construct a WebbPSF gridded PSF model."""

    wfi = roman.WFI()

    wfi.filter = filter
    wfi.detector = detector

    grid = wfi.psf_grid(num_psfs=9, all_detectors=False, oversample=10, save=save, outdir=outdir, outfile="b011.fits")

    return grid


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

    PSF_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../PSFs'))

    if not os.path.exists(PSF_dir):
        os.mkdir(PSF_dir)

    psf_grid = construct_psf_grid(save=True, outdir=PSF_dir)

    print(psf_grid.data.shape)
