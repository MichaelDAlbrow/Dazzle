#
#  Dazzle
#
#   Construct an oversampled representation of an image from dithered undersampled images
#   using forward modelling. ....
#
#
import os

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

__author__ = 'Michael Albrow'

class Image():
    def __init__(self, f_name: str, x_range: tuple = None, y_range: tuple = None):

        if f_name is None:
            raise ValueError("File name must be provided")

        self.f_name = f_name

        with fits.open(f_name) as f:

            self.header = f[0].header
            self.wcs = WCS(f[0].header)

            self.data = f[0].data

            ny, nx = self.data.shape  # Assuming 2D image data

            # Use the provided ranges or default to the full range
            if x_range is None:
                x_range = (0, nx)
            if y_range is None:
                y_range = (0, ny)

            # Validate the ranges
            if not (0 <= x_range[0] < x_range[1] <= nx):
                raise ValueError(f"Invalid x_range: {x_range}")
            if not (0 <= y_range[0] < y_range[1] <= ny):
                raise ValueError(f"Invalid y_range: {y_range}")

            x = np.arange(x_range[0], x_range[1])
            y = np.arange(y_range[0], y_range[1])
            x, y = np.meshgrid(x, y)

            self.coords = self.wcs.pixel_to_world(x, y)


def omoms(x: np.ndarray, order: int) -> np.ndarray:

    match order:
        case 0:
            return 4.0 / 21.0 + (-11.0 / 21.0 + (0.5 - x / 6.0) * x) * x
        case 1:
            return 13.0 / 21.0 + (1.0 / 14.0 + (-1.0 + x / 2.0) * x) * x
        case 2:
            return 4.0 / 21.0 + (3.0 / 7.0 + (0.5 - x / 2.0) * x) * x
        case 3:
            return (1.0 / 42.0 + x * x / 6.0) * x

    raise ValueError("Order must be an integer between 0 and 3.")



if __name__ == '__main__':

    files = [f for f in os.listdir() if "synthpop_test14_t" in f and f.endswith(".fits")]

    xrange = (2000, 2010)
    yrange = (2000, 2010)

    images = [Image(f, xrange, yrange) for f in files]


