import sys
import os
import numpy as np
from astropy.io import fits

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

            self.dx_subpix = f[0].header['DX_SUBPIX']
            self.dy_subpix = f[0].header['DY_SUBPIX']
            self.dx_int = f[0].header['DX_INT']
            self.dy_int = f[0].header['DY_INT']


def read_psf_image(f_name: str) -> np.ndarray:
    with fits.open(f_name) as f:
        data = f["OVERDIST"].data
    return data


def psf_phot(image: Image, psf: np.ndarray, pos: (float, float)) -> float:
    """Fit psf to image at pos. """

    # Evaluate pixel-sampled PSF with correct subpixel offsets

    raise NotImplemented

def extract_lightcurve(images: list[Image], psf: np.ndarray, initial_pos: tuple[float, float]) -> np.ndarray:
    """Do psf photometry on difference-image stack."""

    raise NotImplemented


if __name__ == '__main__':

    start = time.perf_counter()

    files = [f"Results/{f}" for f in os.listdir("Results") if "d_00_" in f and f.endswith(".fits")]
    files.sort()

    images = [Image(f) for f in files]

    print(images[3].dx_subpix)

    psf = read_psf_image("test_psf.fits")

    end = time.perf_counter()
    print(f"Elapsed time: {end - start:0.2f} seconds")
