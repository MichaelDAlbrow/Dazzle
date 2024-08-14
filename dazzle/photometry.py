import sys
import os
import numpy as np
from astropy.io import fits
from photutils.psf import webbpsf_reader, GriddedPSFModel
import time

__author__ = "Michael Albrow"


class Image:

    def __init__(self, f_name: str) -> None:

        # We assume that difference images have been saved with an extension containing the inverse variance,
        # and header fields specifying the integer and subpixel offsets from a reference point.

        if f_name is None:
            raise ValueError("File name must be provided")

        self.f_name = f_name

        with fits.open(f_name) as f:

            self.f_name = f_name
            self.header = f[0].header
            self.data = f[0].data.T
            self.inv_var = f[1].data.T

            self.dx_subpix = f[0].header['DX_SUB']
            self.dy_subpix = f[0].header['DY_SUB']
            self.dx_int = f[0].header['DX_INT']
            self.dy_int = f[0].header['DY_INT']


def read_psf_grid(f_name: str) -> GriddedPSFModel:
    """Read PSF grid created by WebbPSF."""
    grid = webbpsf_reader(f_name)
    return grid


def psf_phot(image: Image, psf: np.ndarray, pos: (float, float)) -> float:
    """Fit psf to image at pos. """

    # Evaluate pixel-sampled PSF with correct subpixel offsets
    # This will include the subpixel part of pos, the offset of this image
    # from the reference, and the centering of the PSF in its array.

    raise NotImplemented


def extract_lightcurve(images: list[Image], psf: np.ndarray, initial_pos: tuple[float, float]) -> np.ndarray:
    """Do psf photometry on difference-image stack."""

    raise NotImplemented


if __name__ == '__main__':

    pass
