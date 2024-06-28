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

            self.x = np.arange(x_range[0], x_range[1])
            self.y = np.arange(y_range[0], y_range[1])
            x, y = np.meshgrid(self.x, self.y)

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


def design_matrix(images: list, ra: np.ndarray, dec: np.ndarray) -> np.ndarray:

    ra_mid = 0.5*(ra[1:]+ra[:-1])
    dec_mid = 0.5*(dec[1:]+dec[:-1])

    n_ra = len(ra_mid)
    n_dec = len(dec_mid)

    ra_width = ra_mid[0] - ra[0]
    dec_width = dec_mid[0] - dec[0]

    print("ra_mid", ra_mid)
    print("dec_mid", dec_mid)
    print("ra_width", ra_width)
    print("dec_width", dec_width)


    p = len(ra_mid)*len(dec_mid)*16
    m = len(images[0].coords.ra.degree)*len(images[0].coords.dec.degree)*len(images)

    A = np.zeros((m, p**2))

    # Iterate over all input image pixels (im, ix, iy).
    # k is the row number.
    # jra and jdec are the indices of this pixel in (ra, dec) space.

    k = 0
    for im in images:
        for ix in range(len(im.x)):

            for iy in range(len(im.y)):

                print('ra')
                print(ra)
                print('dec')
                print(dec)
                print('im ra dec')
                print(im.coords[ix,iy].ra.degree, im.coords[ix, iy].dec.degree)

                jra = np.where((ra_mid - ra_width < im.coords[ix, iy].ra.degree) &
                                       (ra_mid + ra_width > im.coords[ix, iy].ra.degree))[0]
                jdec = np.where((dec_mid - dec_width < im.coords[ix, iy].dec.degree) &
                                     (dec_mid + dec_width > im.coords[ix, iy].dec.degree))[0]

                if len(jra) > 0 and len(jdec) > 0:

                    jra = jra[0]
                    jdec = jdec[0]

                    # 16 non-zero basis functions for each row
                    for ord_x in range(4):
                        xp = omoms(im.coords[ix, iy].ra.degree-ra_mid[jra], ord_x)
                        for ord_y in range(4):
                            A[k, 16*(jra*n_dec+jdec)+4*ord_x+ord_y] = (
                                    xp * omoms(im.coords[ix, iy].dec.degree-dec_mid[jdec], ord_y))
                k += 1

    return A



if __name__ == '__main__':

    files = [f for f in os.listdir() if "synthpop_test14_t" in f and f.endswith(".fits")]

    xrange = (2000, 2010)
    yrange = (2000, 2010)

    images = [Image(f, xrange, yrange) for f in files]

    ra = images[0].coords.ra.degree
    dec = images[0].coords.dec.degree

    ra_range = (np.min(ra), np.max(ra))
    dec_range = (np.min(dec), np.max(dec))

    print("Coordinate ranges:", ra_range, dec_range)

    # Divide ra and dec ranges into n_pixel divisions
    ra_out = np.linspace(*ra_range, num=(xrange[1]-xrange[0]+1), endpoint=True)
    dec_out = np.linspace(*dec_range, num=(yrange[1]-yrange[0]+1), endpoint=True)
    ra_mid = 0.5*(ra_out[1:]+ra_out[:-1])
    dec_mid = 0.5*(dec_out[1:]+dec_out[:-1])

    print("Coordinate output vectors:", ra_mid, dec_mid)

    A = design_matrix(images, ra_out, dec_out)

    print(A.shape)




