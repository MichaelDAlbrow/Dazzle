#
#  Dazzle
#
#   Construct an oversampled representation of an image from dithered undersampled images
#   using forward modelling. ....
#
#

import sys
import os
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from scipy.ndimage import minimum_filter

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

#from concurrent.futures import ProcessPoolExecutor

__author__ = "Michael Albrow"

POLY_ORDER = 5  # i.e. 5 coefficients


class Image:

    def __init__(self, f_name: str, x_range: tuple = None, y_range: tuple = None) -> None:

        if f_name is None:
            raise ValueError("File name must be provided")

        self.f_name = f_name

        with fits.open(f_name) as f:

            self.f_name = f_name
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

            self.data = f[0].data[x_range[0]:x_range[1], y_range[0]:y_range[1]].T
            self.sigma = f[1].data[x_range[0]:x_range[1], y_range[0]:y_range[1]].T
            self.mask = np.ones_like(self.data)
            self.mask[self.data < 0] = 0.0

            self.inv_var = self.mask / self.sigma ** 2
            self.inv_var[np.isnan(self.inv_var)] = 0.0

        self.x = np.arange(x_range[0], x_range[1])
        self.y = np.arange(y_range[0], y_range[1])

        self.difference = None
        self.dRdx = None
        self.dRdy = None

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

        self.dx_int = np.rint(dx).astype(int)
        self.dy_int = np.rint(dy).astype(int)
        self.dx_subpix = (dx - self.dx_int)
        self.dy_subpix = (dy - self.dy_int)

        print(self.f_name, dx, dy)

        return np.array([dx, dy])


def legendre(x: float | np.ndarray, order: int) -> float | np.ndarray:
    """Shifted legendre functions of given order evaluated at x."""

    match order:
        case 0:
            return x * 0.0 + 1.0
        case 1:
            return 2.0 * x - 1.0
        case 2:
            return 6.0 * x ** 2 - 6.0 * x + 1.0
        case 3:
            return 20.0 * x ** 3 - 30.0 * x ** 2 + 12.0 * x - 1.0
        case 4:
            return 70.0 * x ** 4 - 140.0 * x ** 3 + 90.0 * x ** 2 - 20.0 * x + 1.0
        case 5:
            return 252.0 * x ** 5 - 630.0 * x ** 4 + 560.0 * x ** 3 - 210.0 * x ** 2 + 30.0 * x - 1.0

    raise ValueError("Order must be an integer between 0 and 5.")


def grad_legendre(x: float | np.ndarray, order: int) -> float | np.ndarray:
    """Gradient of shifted legendre functions of given order evaluated at x."""

    match order:
        case 0:
            return 0.0
        case 1:
            return 2.0
        case 2:
            return 12.0 * x - 6.0
        case 3:
            return 60.0 * x ** 2 - 60.0 * x + 12.0
        case 4:
            return 280.0 * x ** 3 - 420.0 * x ** 2 + 180.0 * x - 20.0
        case 5:
            return 1260.0 * x ** 4 - 2520.0 * x ** 3 + 1680.0 * x ** 2 - 4200 * x + 30.0

    raise ValueError("Order must be an integer between 0 and 5.")


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


def grad_omoms(x: float | np.ndarray, order: int) -> float | np.ndarray:
    """Derivatives of cubic O-MOMS polynomials of given order evaluated at x."""

    match order:
        case 0:
            return -11.0 / 21.0 + (1.0 - x / 2.0) * x
        case 1:
            return 1.0 / 14.0 + (-2.0 + 1.5 * x) * x
        case 2:
            return 3.0 / 7.0 + (1.0 - 1.5 * x) * x
        case 3:
            return 1.0 / 42.0 + x * x / 2.0

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
        c[k] = im.inv_var[i + im.dx_int, j + im.dy_int]
    return c


def design_matrix(images: list[Image]) -> np.ndarray:
    """Compute the design matrix for bicubic legendre basis functions, given a list of images with
    pixel offsets from a reference."""

    m = len(images)

    A = np.zeros((m, POLY_ORDER ** 2))

    for k, im in enumerate(images):

        for ord_x in range(POLY_ORDER):
            xp = legendre(im.dx_subpix, ord_x)
            for ord_y in range(POLY_ORDER):
                A[k, POLY_ORDER * ord_x + ord_y] = xp * legendre(im.dy_subpix, ord_y)

    return A


def solve_linear(images: list[Image], xrange: tuple, yrange: tuple, reference_image_range: tuple = None,
                 save: bool = True, output_file: str = "coefficients.fits") -> np.ndarray:
    """For each pixel, set up and solve the system of linear equations to compute the basis coefficients."""

    def solve_linear_pixel(i: int, j: int) -> (int, int, np.ndarray):
        """Solve linear equations to compute basis-function coefficients for one pixel."""

        y = data_vector(images[start:end], i, j)
        C_inv = inverse_variance_vector(images[start:end], i, j)

        A = np.dot(X.T * C_inv, X)
        try:
            B = np.linalg.solve(A, np.identity(A.shape[0]))
            return i, j, np.dot(B, np.dot(X.T, y * C_inv)).reshape(POLY_ORDER, POLY_ORDER)
        except np.linalg.LinAlgError:
            return i, j, np.zeros((POLY_ORDER, POLY_ORDER))

    if reference_image_range is not None:
        start = reference_image_range[0]
        end = reference_image_range[1]
    else:
        start = 0
        end = len(images)

    X = design_matrix(images[start:end])

    nx = xrange[1] - xrange[0]
    ny = yrange[1] - yrange[0]
    result = np.zeros((nx, ny, POLY_ORDER, POLY_ORDER))

    # with ProcessPoolExecutor(max_workers=16) as executor:
    #     futures = [executor.submit(solve_linear_pixel, i, j) for i in range(xrange[0], xrange[1]) for j in
    #                range(yrange[0], yrange[1])]
    #
    #     for future in futures:
    #         i, j, res = future.result()
    #         result[i - xrange[0], j - yrange[0], :] = res

    for i in range(xrange[0], xrange[1]):
        for j in range(yrange[0], yrange[1]):
            y = data_vector(images[start:end], i, j)
            C_inv = inverse_variance_vector(images[start:end], i, j)

            A = np.dot(X.T * C_inv, X)
            try:
                B = np.linalg.solve(A, np.identity(A.shape[0]))
            except np.linalg.LinAlgError:
                continue

            result[i - xrange[0], j - yrange[0], :] = np.dot(B, np.dot(X.T, y * C_inv)).reshape(POLY_ORDER, POLY_ORDER)

    if save:
        hdu = fits.PrimaryHDU(result)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(output_file, overwrite=True)

    return result


def evaluate_bicubic_legendre(theta, xrange: tuple, yrange: tuple, oversample_ratio=10) -> np.ndarray:
    """Evaluate the bicubic legendre function with coefficients theta over xrange, yrange."""

    nx = xrange[1] - xrange[0]
    ny = yrange[1] - yrange[0]
    z = np.zeros((nx * oversample_ratio, ny * oversample_ratio))

    x = np.linspace(0.5, -0.5, oversample_ratio + 1)[:-1]
    y = np.linspace(0.5, -0.5, oversample_ratio + 1)[:-1]

    yp = np.empty((oversample_ratio, POLY_ORDER))
    xp = np.empty((oversample_ratio, POLY_ORDER))

    for order in range(POLY_ORDER):
        yp[:, order] = legendre(y, order)
        xp[:, order] = legendre(x, order)

    for i in range(nx):
        for j in range(ny):
            z[i * oversample_ratio:(i + 1) * oversample_ratio, j * oversample_ratio:(j + 1) * oversample_ratio] = (
                np.einsum("ml,rl,sm->sr", theta[i, j], xp, yp))

    return z


def make_difference_images(images: list[Image], theta: np.ndarray, xrange: tuple, yrange: tuple,
                           output_dir: str = "Results", iteration: int = 0) -> None:
    """Construct and save difference images."""

    try:
        os.mkdir(output_dir)
    except FileExistsError:
        pass

    xp = np.empty(POLY_ORDER)
    yp = np.empty(POLY_ORDER)

    xp_grad = np.zeros(POLY_ORDER)
    yp_grad = np.zeros(POLY_ORDER)

    for im in images:

        for order in range(POLY_ORDER):
            xp[order] = legendre(im.dx_subpix, order)
            yp[order] = legendre(im.dy_subpix, order)
            xp_grad[order] = grad_legendre(im.dx_subpix, order)
            yp_grad[order] = grad_legendre(im.dy_subpix, order)

        basis = np.zeros((POLY_ORDER, POLY_ORDER))
        for ord_x in range(POLY_ORDER):
            for ord_y in range(POLY_ORDER):
                basis[ord_x, ord_y] = xp[ord_x] * yp[ord_y]

        im.model = np.zeros_like(im.data)
        im.dRdx = np.zeros_like(im.data)
        im.dRdy = np.zeros_like(im.data)

        im.model[xrange[0] + im.dx_int:xrange[1] + im.dx_int, yrange[0] + im.dy_int:yrange[1] + im.dy_int] = (
            np.einsum("ijml,m,l", theta, xp, yp))
        im.dRdx[xrange[0] + im.dx_int:xrange[1] + im.dx_int, yrange[0] + im.dy_int:yrange[1] + im.dy_int] = (
            np.einsum("ijml,m,l", theta, xp_grad, yp))
        im.dRdy[xrange[0] + im.dx_int:xrange[1] + im.dx_int, yrange[0] + im.dy_int:yrange[1] + im.dy_int] = (
            np.einsum("ijml,m,l", theta, xp, yp_grad))

        # For now, we write out lots of information. This shouldn't be necessary in the future.

        prefix = f"d_{iteration:02d}_"
        hdu = fits.PrimaryHDU((im.data - im.model).T)
        hdu.writeto(f"{output_dir}/{prefix}{os.path.basename(im.f_name)}", overwrite=True)

        prefix = f"z_{iteration:02d}_"
        hdu = fits.PrimaryHDU(im.model.T)
        hdu.writeto(f"{output_dir}/{prefix}{os.path.basename(im.f_name)}", overwrite=True)

        prefix = f"r_{iteration:02d}_"
        r = im.data - im.model
        r[np.isnan(r)] = 0.0
        im.difference = r
        hdu = fits.PrimaryHDU(r.T)
        hdu.writeto(f"{output_dir}/{prefix}{os.path.basename(im.f_name)}", overwrite=True)

        prefix = f"rx_{iteration:02d}_"
        hdu = fits.PrimaryHDU(im.dRdx.T)
        hdu.writeto(f"{output_dir}/{prefix}{os.path.basename(im.f_name)}", overwrite=True)

        prefix = f"ry_{iteration:02d}_"
        hdu = fits.PrimaryHDU(im.dRdy.T)
        hdu.writeto(f"{output_dir}/{prefix}{os.path.basename(im.f_name)}", overwrite=True)

        prefix = f"a_{iteration:02d}_"
        hdu = fits.PrimaryHDU(im.data.T)
        hdu.writeto(f"{output_dir}/{prefix}{os.path.basename(im.f_name)}", overwrite=True)

        prefix = f"e_{iteration:02d}_"
        r = im.mask.T * r.T / im.sigma.T
        r[np.isnan(r)] = 0.0
        hdu = fits.PrimaryHDU(r)
        hdu.writeto(f"{output_dir}/{prefix}{os.path.basename(im.f_name)}", overwrite=True)

        prefix = f"m_{iteration:02d}_"
        hdu = fits.PrimaryHDU(im.mask.T)
        hdu.writeto(f"{output_dir}/{prefix}{os.path.basename(im.f_name)}", overwrite=True)

        prefix = f"iv_{iteration:02d}_"
        hdu = fits.PrimaryHDU(im.inv_var.T)
        hdu.writeto(f"{output_dir}/{prefix}{os.path.basename(im.f_name)}", overwrite=True)


def refine_offsets(images: list[Image], xrange: tuple, yrange: tuple) -> np.ndarray:
    """Computes changes to the offsets defined for each image from the astrometric reference."""

    delta_xy = np.zeros((len(images), 2))

    A = np.zeros((2, 2))
    b = np.zeros(2)

    for k, im in enumerate(images):

        # im.inv_var is already set to zero for saturated pixels.
        # For this function we don't want to count adjacent pixels.

        iv = minimum_filter(im.inv_var, size=3, mode="constant", cval=0.0)

        A[0, 0] = np.sum(im.dRdx ** 2 * iv)
        A[0, 1] = np.sum(im.dRdx * im.dRdy * iv)
        A[1, 1] = np.sum(im.dRdy ** 2 * iv)
        A[1, 0] = A[0, 1]

        b[0] = -np.sum(im.difference * im.dRdx * iv)
        b[1] = -np.sum(im.difference * im.dRdy * iv)

        delta_xy[k, :] = np.linalg.solve(A, b)
        im.dx_subpix -= delta_xy[k, 0]
        im.dy_subpix -= delta_xy[k, 1]

    return delta_xy


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

    files = [f"Data/test18/{f}" for f in os.listdir("Data/test18") if "synthpop_test18_t" in f and f.endswith(".fits")]
    files.sort()

    #input_yrange = (2080, 2100)
    #input_xrange = (2090, 2110)
    input_yrange = (1000, 3000)
    input_xrange = (1000, 3000)

    n_input_images = len(files)
    reference_image_range = (0, 96)

    images = [Image(f, input_xrange, input_yrange) for f in files[:n_input_images]]

    # Compute the offset in pixels between each image and the first one.

    offsets = np.zeros((len(images), 2))
    for k, im in enumerate(images):
        offsets[k, :] = im.compute_offset(images[0])

    print("offsets standard deviation:", np.std(offsets[:, 0]), np.std(offsets[:, 1]))
    plot_offsets(offsets)

    output_xrange = (-np.rint(np.min(offsets[:, 0])).astype(int),
                     images[0].data.shape[0] - np.rint(np.max(offsets[:, 0])).astype(int) - 1)
    output_yrange = (-np.rint(np.min(offsets[:, 1])).astype(int),
                     images[0].data.shape[1] - np.rint(np.max(offsets[:, 1])).astype(int) - 1)

    print("Input ranges:", input_xrange, input_yrange)
    print("Output ranges:", output_xrange, output_yrange)

    end = time.perf_counter()
    print(f"Elapsed time: {end - start:0.2f} seconds")

    # test refine offsets
    #images[-1].dx_subpix += 0.05
    #images[-1].dy_subpix += 0.1

    for iter in range(1):

        if iter == 0:

            # Compute the coefficients of the basis functions for each image pixel
            print("Computing coefficients ...")
            theta = solve_linear(images, output_xrange, output_yrange, reference_image_range=reference_image_range)
            end = time.perf_counter()
            print(f"Elapsed time: {end - start:0.2f} seconds")

            # Compute and save an oversampled image
            print("Computing oversampled image ...")
            z = evaluate_bicubic_legendre(theta, output_xrange, output_yrange)
            end = time.perf_counter()
            print(f"Elapsed time: {end - start:0.2f} seconds")

            print("Writing oversampled image ...")
            hdu = fits.PrimaryHDU(z.T)
            hdu.writeto(f"Results/test18_oversampled_{iter:02d}.fits", overwrite=True)
            end = time.perf_counter()
            print(f"Elapsed time: {end - start:0.2f} seconds")

        # Difference images
        print("Making difference images ...")
        make_difference_images(images, theta, output_xrange, output_yrange, iteration=iter)
        end = time.perf_counter()
        print(f"Elapsed time: {end - start:0.2f} seconds")

        # Refining offsets
        print("Refining offsets ...")
        refine_offsets(images, output_xrange, output_yrange)
        end = time.perf_counter()
        print(f"Elapsed time: {end - start:0.2f} seconds")