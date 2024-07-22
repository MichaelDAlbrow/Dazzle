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

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

__author__ = "Michael Albrow"


class Image:

    def __init__(self, f_name: str, x_range: tuple = None, y_range: tuple = None) -> None:

        if f_name is None:
            raise ValueError("File name must be provided")

        self.f_name = f_name

        with fits.open(f_name) as f:

            self.header = f[0].header
            self.wcs = WCS(f[0].header)

            ny, nx = f[0].data.shape  # Assuming 2D image data

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

            self.data = f[0].data[x_range[0]:x_range[1], y_range[0]:y_range[1]]
            self.sigma = f[1].data[x_range[0]:x_range[1], y_range[0]:y_range[1]]
            self.mask = 1 - f[2].data[x_range[0]:x_range[1], y_range[0]:y_range[1]]
            self.mask[self.mask < 0] = 0

        self.x = np.arange(x_range[0], x_range[1])
        self.y = np.arange(y_range[0], y_range[1])

        self.dy_subpix = None
        self.dx_subpix = None
        self.dy_int = None
        self.dx_int = None

    def compute_offset(self, ref_im: 'Image') -> np.ndarray:
        """Compute the integer and subpixel offsets from the reference image using the WCS."""
        x1_ref, y1_ref = 1, 1
        c = self.wcs.pixel_to_world(x1_ref, y1_ref)
        x2_ref, y2_ref = ref_im.wcs.world_to_pixel(c)
        dx = x1_ref - x2_ref
        dy = y1_ref - y2_ref
        self.dx_int = np.rint(dy).astype(int)
        self.dy_int = np.rint(dx).astype(int)
        self.dx_subpix = (dy - self.dx_int)
        self.dy_subpix = (dx - self.dy_int)
        print(self.f_name, dx, dy)
        return np.array([dx, dy])


def omoms(x: float | np.ndarray, order: int) -> float | np.ndarray:
    """Cubic O-MOMS polynomials of given order evaluated at x."""

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


def data_vector(images: list[Image], i: int, j: int) -> np.ndarray:
    """Compute the vector of all image data values for pixels i, j."""

    m = len(images)
    y = np.zeros(m)

    for k, im in enumerate(images):
        try:
            y[k] = im.data[i + im.dx_int, j + im.dy_int]
        except IndexError:
            print("IndexError", i, im.dx_int, j, im.dy_int)
            raise

    return y


def inverse_variance_vector(images: list[Image], i: int, j: int) -> np.ndarray:
    """Compute the inverse variance vector for all image pixels."""

    m = len(images)
    c = np.zeros(m)

    for k, im in enumerate(images):
        c[k] = im.mask[i + im.dx_int, j + im.dy_int] / im.sigma[i + im.dx_int, j + im.dy_int] ** 2
    return c


def design_matrix(images: list[Image]) -> np.ndarray:
    """Compute the design matrix for bicubic OMOMS basis functions, given a list of images with
    pixel offsets from a reference."""

    p = 4
    m = len(images)

    A = np.zeros((m, p ** 2))

    for k, im in enumerate(images):

        for ord_x in range(4):
            xp = omoms(im.dx_subpix, ord_x)
            for ord_y in range(4):
                A[k, 4 * ord_x + ord_y] = xp * omoms(im.dy_subpix, ord_y)

    return A


def solve_linear(images: list[Image], xrange: tuple, yrange: tuple) -> np.ndarray:
    """For each pixel, set up and solve the system of linear equations to compute the basis coefficients."""

    X = design_matrix(images)

    nx = xrange[1] - xrange[0]
    ny = yrange[1] - yrange[0]
    result = np.zeros((nx, ny, 16))

    for i in range(xrange[0], xrange[1]):
        for j in range(yrange[0], yrange[1]):
            y = data_vector(images, i, j)
            C_inv = inverse_variance_vector(images, i, j)

            A = np.dot(X.T * C_inv, X)
            try:
                B = np.linalg.solve(A, np.identity(A.shape[0]))
            except np.linalg.LinAlgError:
                continue

            result[i - xrange[0], j - yrange[0], :] = np.dot(B, np.dot(X.T, y * C_inv))

    return result


def evaluate_bicubic_omoms(theta, xrange: tuple, yrange: tuple, oversample_ratio=10) -> np.ndarray:
    """Evaluate the bicubic omoms function with coefficients theta over xrange, yrange."""

    nx = xrange[1] - xrange[0]
    ny = yrange[1] - yrange[0]
    z = np.zeros((nx * oversample_ratio, ny * oversample_ratio))

    x = np.linspace(-0.5, 0.5, oversample_ratio)
    y = np.linspace(-0.5, 0.5, oversample_ratio)
    yy, xx = np.meshgrid(-x, -y)

    for i in range(nx):
        for j in range(ny):
            zp = np.zeros((oversample_ratio, oversample_ratio))
            for ord_x in range(4):
                xp = omoms(xx, ord_x)
                for ord_y in range(4):
                    zp += xp * omoms(yy, ord_y) * theta[i, j, 4 * ord_x + ord_y]
            z[i * oversample_ratio:(i + 1) * oversample_ratio, j * oversample_ratio:(j + 1) * oversample_ratio] = zp

    return z


def make_difference_images(images: list[Image], theta: np.ndarray, xrange: tuple, yrange: tuple,
                           output_dir: str = "Results", prefix: str = "d_") -> None:
    """Construct and save difference images."""
    try:
        os.mkdir(output_dir)
    except FileExistsError:
        pass

    for im in images:

        # Basis functions evaluated at image offset
        basis = np.zeros(16)
        for ord_x in range(4):
            xp = omoms(im.dx_subpix, ord_x)
            for ord_y in range(4):
                basis[4*ord_x + ord_y] = xp * omoms(im.dy_subpix, ord_y)

        z = np.zeros_like(im.data)
        for i in range(xrange[0], xrange[1]):
            for j in range(yrange[0], yrange[1]):
                z[i+im.dx_int, j+im.dy_int] = np.dot(basis, theta[i-xrange[0], j-yrange[0]])

        # z is still the sampled image, not the difference image

        hdu = fits.PrimaryHDU(im.data - z)
        hdu.writeto(f"{output_dir}/{prefix}{os.path.basename(im.f_name)}", overwrite=True)


def plot_offsets(offsets: np.ndarray) -> None:
    """Plot the offsets."""
    fig, ax = plt.subplots(2, 2, figsize=(11, 11))
    ax[0, 0].scatter(offsets[:, 0], offsets[:, 1], marker="o", s=5)
    ax[0, 0].set_xlabel("dx")
    ax[0, 0].set_ylabel("dy")
    ax[0, 1].scatter(offsets[:, 0] - np.rint(offsets[:, 0]), offsets[:, 1] - np.rint(offsets[:, 1]), marker="o", s=5)
    ax[0, 1].set_xlabel("dx")
    ax[0, 1].set_ylabel("dy")
    ax[1, 0].hist(np.rint(offsets[:, 0]), bins=np.linspace(-3.5, 3.5, 8))
    ax[1, 0].set_xlabel("dx")
    ax[1, 0].set_ylabel("N")
    ax[1, 1].hist(np.rint(offsets[:, 1]), bins=np.linspace(-3.5, 3.5, 8))
    ax[1, 1].set_xlabel("dy")
    ax[1, 1].set_ylabel("N")
    plt.savefig("offsets.png")


if __name__ == '__main__':

    start = time.perf_counter()

    files = [f"Data/{f}" for f in os.listdir("Data") if "synthpop_test16_t" in f and f.endswith(".fits")]
    files.sort()

    input_yrange = (1680, 2340)
    input_xrange = (1980, 2640)

    n_input_images = 20

    images = [Image(f, input_xrange, input_yrange) for f in files[:n_input_images]]

    # Compute the offset in pixels between each image and the first one.
    offsets = np.zeros((len(images), 2))
    for k, im in enumerate(images):
        offsets[k, :] = im.compute_offset(images[0])
    print("offsets standard deviation:", np.std(offsets[:, 0]), np.std(offsets[:, 1]))
    plot_offsets(offsets)
    output_yrange = (-int(np.min(offsets[:, 0])), images[0].data.shape[0]-int(np.max(offsets[:, 0]))-1)
    output_xrange = (-int(np.min(offsets[:, 1])), images[0].data.shape[1]-int(np.max(offsets[:, 1]))-1)

    # Compute the coefficients of the basis functions for each image pixel
    theta = solve_linear(images, output_xrange, output_yrange)

    # Compute and save an oversampled image
    z = evaluate_bicubic_omoms(theta, output_xrange, output_yrange)
    hdu = fits.PrimaryHDU(z)
    hdu.writeto("oversampled.fits", overwrite=True)

    # Difference images
    make_difference_images(images, theta, output_xrange, output_yrange)

    end = time.perf_counter()
    print(f"Elapsed time: {end-start:0.2f} seconds")
