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

from scipy.ndimage import minimum_filter
from scipy.signal import convolve2d

from .utils import write_as_fits

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

__author__ = "Michael Albrow"

POLY_ORDER = 5  # i.e. 5 coefficients


class Image:

    def __init__(self, f_name: str, x_range: tuple = None, y_range: tuple = None, debug: bool = False) -> None:

        if f_name is None:
            raise ValueError("File name must be provided")

        self.f_name = f_name

        with fits.open(f_name) as f:

            if debug:
                print(f"Reading {f_name}")
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
            self.mask = np.ones_like(self.data, dtype=bool)
            self.mask[self.data < 0] = 0
            self.vmask = np.ones_like(self.data, dtype=bool)

            self.inv_var = self.mask / self.sigma ** 2
            self.inv_var[np.isnan(self.inv_var)] = 0.0

        self.x = np.arange(x_range[0], x_range[1])
        self.y = np.arange(y_range[0], y_range[1])

        self.model = None
        self.noise = None
        self.vmask_delta = None
        self.difference = None
        self.dRdx = None
        self.dRdy = None

        self.dy_subpix = None
        self.dx_subpix = None
        self.dy_int = None
        self.dx_int = None

    def compute_offset(self, ref_im: 'Image', debug: bool = False) -> np.ndarray:
        """Compute the integer and subpixel offsets from the reference image using the WCS."""

        x1_ref, y1_ref = 1, 1
        c = self.wcs.pixel_to_world(x1_ref, y1_ref)
        x2_ref, y2_ref = ref_im.wcs.world_to_pixel(c)  # Do we want world_to_array_index() here instead?
        dx = x1_ref - x2_ref
        dy = y1_ref - y2_ref

        self.dx_int = np.rint(dx).astype(int)
        self.dy_int = np.rint(dy).astype(int)
        self.dx_subpix = (dx - self.dx_int)
        self.dy_subpix = (dy - self.dy_int)

        if debug:
            print(self.f_name, dx, dy)

        return np.array([dx, dy])


def basis_function(x: float | np.ndarray, order: int) -> float | np.ndarray:
    return legendre(x, order)


def grad_basis_function(x: np.ndarray, order: int) -> float | np.ndarray:
    return grad_legendre(x, order)


def legendre(x: float | np.ndarray, order: int) -> float | np.ndarray:
    """
    Shifted legendre functions of given order evaluated at x.
   """

    x = x + 0.5

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
    """
    Gradient of shifted legendre functions of given order evaluated at x.
    """

    x = x + 0.5

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

    x = 3 * x + 0.5

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

    x = 3 * x + 0.5

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


def data_matrix(images: list[Image], xrange: (int, int), yrange: (int, int), expand_factor: float = 1.0) -> np.ndarray:
    """Compute the vector of all image data values for pixels i, j."""

    n = len(images)
    m = n * 6

    nx = xrange[1] - xrange[0]
    ny = yrange[1] - yrange[0]

    z = np.zeros((nx, ny, m))

    counter = 0
    for im in images:
        i0 = im.dx_int + xrange[0]
        j0 = im.dy_int + yrange[0]
        z[:, :, counter] = im.data[i0:i0 + nx, j0:j0 + ny]
        counter += 1
        if im.dx_subpix < 0.5 * expand_factor - 1.0:
            z[:, :, counter] = im.data[i0 - 1:i0 + nx - 1, j0:j0 + ny]
            counter += 1
        if im.dx_subpix > -0.5 * expand_factor + 1.0:
            z[:, :, counter] = im.data[i0 + 1:i0 + nx + 1, j0:j0 + ny]
            counter += 1
        if im.dy_subpix < 0.5 * expand_factor - 1.0:
            z[:, :, counter] = im.data[i0:i0 + nx, j0 - 1:j0 + ny - 1]
            counter += 1
        if im.dy_subpix > -0.5 * expand_factor + 1.0:
            z[:, :, counter] = im.data[i0:i0 + nx, j0 + 1:j0 + ny + 1]
            counter += 1

    z = z[:, :, :counter]

    return z


def inverse_variance_matrix(images: list[Image], xrange: (int, int), yrange: (int, int),
                            expand_factor: float = 1.0) -> np.ndarray:
    """Compute the inverse variance vector for all image pixels."""

    m = len(images) * 6

    nx = xrange[1] - xrange[0]
    ny = yrange[1] - yrange[0]

    c = np.zeros((nx, ny, m))

    counter = 0
    for im in images:
        i0 = im.dx_int + xrange[0]
        j0 = im.dy_int + yrange[0]
        c[:, :, counter] = (im.inv_var[i0:i0 + nx, j0:j0 + ny] *
                            im.vmask[i0:i0 + nx, j0:j0 + ny])
        counter += 1
        if im.dx_subpix < 0.5 * expand_factor - 1.0:
            c[:, :, counter] = (im.inv_var[i0 - 1:i0 + nx - 1, j0:j0 + ny] *
                                im.vmask[i0 - 1:i0 + nx - 1, j0:j0 + ny])
            counter += 1
        if im.dx_subpix > -0.5 * expand_factor + 1.0:
            c[:, :, counter] = (im.inv_var[i0 + 1:i0 + nx + 1, j0:j0 + ny] *
                                im.vmask[i0 + 1:i0 + nx + 1, j0:j0 + ny])
            counter += 1
        if im.dy_subpix < 0.5 * expand_factor - 1.0:
            c[:, :, counter] = (im.inv_var[i0:i0 + nx, j0 - 1:j0 + ny - 1] *
                                im.vmask[i0:i0 + nx, j0 - 1:j0 + ny - 1])
            counter += 1
        if im.dy_subpix > -0.5 * expand_factor + 1.0:
            c[:, :, counter] = (im.inv_var[i0:i0 + nx, j0 + 1:j0 + ny + 1] *
                                im.vmask[i0:i0 + nx, j0 + 1:j0 + ny + 1])
            counter += 1

    c = c[:, :, :counter]

    return c


def design_matrix(images: list[Image], expand_factor: float = 1.0) -> np.ndarray:
    """Compute the design matrix for bi-poly basis functions, given a list of images with
    pixel offsets from a reference."""

    dx = []
    dy = []
    for k, im in enumerate(images):
        dx.append(im.dx_subpix / expand_factor)
        dy.append(im.dy_subpix / expand_factor)
        if im.dx_subpix < 0.5 * expand_factor - 1.0:
            dx.append((im.dx_subpix + 1.0) / expand_factor)
            dy.append(im.dy_subpix / expand_factor)
        if im.dx_subpix > -0.5 * expand_factor + 1.0:
            dx.append((im.dx_subpix - 1.0) / expand_factor)
            dy.append(im.dy_subpix / expand_factor)
        if im.dy_subpix < 0.5 * expand_factor - 1.0:
            dx.append(im.dx_subpix / expand_factor)
            dy.append((im.dy_subpix + 1.0) / expand_factor)
        if im.dy_subpix > -0.5 * expand_factor + 1.0:
            dx.append(im.dx_subpix / expand_factor)
            dy.append((im.dy_subpix - 1.0) / expand_factor)

    dx = np.array(dx)
    dy = np.array(dy)
    m = len(dx)

    xp = np.zeros((m, POLY_ORDER))
    yp = np.zeros((m, POLY_ORDER))

    for order in range(POLY_ORDER):
        xp[:, order] = basis_function(dx, order)
        yp[:, order] = basis_function(dy, order)

    A = np.zeros((m, POLY_ORDER ** 2))

    for k in range(m):
        for ord_x in range(POLY_ORDER):
            for ord_y in range(POLY_ORDER):
                A[k, POLY_ORDER * ord_x + ord_y] = xp[k, ord_x] * yp[k, ord_y]

    return A


def solve_linear(images: list[Image], xrange: tuple, yrange: tuple, reference_image_range: tuple = None,
                 expand_factor: float = 1.0, save: bool = True, output_dir: str = "tests",
                 output_file: str = "coefficients.fits", theta: np.ndarray = None) -> np.ndarray:
    """For each pixel, set up and solve the system of linear equations to compute the basis coefficients."""

    if reference_image_range is not None:
        start = reference_image_range[0]
        end = reference_image_range[1]
    else:
        start = 0
        end = len(images)

    X = design_matrix(images[start:end], expand_factor=expand_factor)
    y = data_matrix(images[start:end], xrange, yrange, expand_factor=expand_factor)
    C_inv = inverse_variance_matrix(images[start:end], xrange, yrange, expand_factor=expand_factor)

    nx = xrange[1] - xrange[0]
    ny = yrange[1] - yrange[0]

    # We use compute_mask to only re-solve for pixels that have changed their vmask and their neighbours

    if theta is not None:

        compute_mask = np.zeros((nx, ny), dtype=bool)
        result = theta
        m = np.ones((3, 3)) / 5
        m[0, 0] = 0
        m[0, 2] = 0
        m[2, 0] = 0
        m[2, 2] = 0

        for im in images:

            i0 = im.dx_int + xrange[0]
            j0 = im.dy_int + yrange[0]

            if im.vmask_delta is None:

                compute_mask = np.ones_like(im.data, dtype=bool)

            else:

                compute_mask_im = convolve2d(im.vmask_delta[i0:i0 + nx, j0:j0 + ny], m, mode="same", boundary="fill",
                                             fillvalue=1)
                compute_mask[compute_mask_im > 0.01] = 1

    else:

        compute_mask = np.ones((nx, ny), dtype=bool)
        result = np.zeros((nx, ny, POLY_ORDER, POLY_ORDER))

    for i in range(nx):
        for j in range(ny):

            if compute_mask[i, j]:

                A = np.dot(X.T * C_inv[i, j, :], X)

                try:
                    B = np.linalg.solve(A, np.identity(A.shape[0]))
                except np.linalg.LinAlgError:
                    continue

                result[i, j, :] = np.dot(B, np.dot(X.T, y[i, j, :] * C_inv[i, j, :])).reshape(POLY_ORDER, POLY_ORDER)

    if save:
        write_as_fits(f"{output_dir}/{output_file}", result)

    return result


def evaluate_bipolynomial_basis(theta, xrange: tuple, yrange: tuple, oversample_ratio=10,
                                expand_factor: float = 1.0) -> np.ndarray:
    """Evaluate the bi-polynomial basis function with coefficients theta over xrange, yrange."""

    nx = xrange[1] - xrange[0]
    ny = yrange[1] - yrange[0]
    z = np.zeros((nx * oversample_ratio, ny * oversample_ratio))

    x = np.linspace(0.5, -0.5, oversample_ratio + 1)[:-1] / expand_factor
    y = np.linspace(0.5, -0.5, oversample_ratio + 1)[:-1] / expand_factor

    yp = np.empty((oversample_ratio, POLY_ORDER))
    xp = np.empty((oversample_ratio, POLY_ORDER))

    for order in range(POLY_ORDER):
        yp[:, order] = basis_function(y, order)
        xp[:, order] = basis_function(x, order)

    for i in range(nx):
        for j in range(ny):
            z[i * oversample_ratio:(i + 1) * oversample_ratio, j * oversample_ratio:(j + 1) * oversample_ratio] = (
                np.einsum("ml,rl,sm->sr", theta[i, j], xp, yp))

    return z


def make_difference_images(images: list[Image], theta: np.ndarray, xrange: tuple, yrange: tuple,
                           output_dir: str = "Results", iteration: int = 0, expand_factor: float = 1.0,
                           write_debug_images: bool = False, write_difference_images: bool = True) -> None:
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
            xp[order] = basis_function(im.dx_subpix / expand_factor, order)
            yp[order] = basis_function(im.dy_subpix / expand_factor, order)
            xp_grad[order] = grad_basis_function(im.dx_subpix / expand_factor, order)
            yp_grad[order] = grad_basis_function(im.dy_subpix / expand_factor, order)

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

        # Update mask to include edge pixels
        im.mask[:xrange[0] + 2, :] = 0
        im.mask[xrange[1] - 3:, :] = 0
        im.mask[:, :yrange[0] + 2] = 0
        im.mask[:, yrange[1] - 3:] = 0

        # difference image
        r = im.data - im.model
        r[np.isnan(r)] = 0.0
        im.difference = r

        # For now, we write out lots of information. This shouldn't be necessary in the future.

        if write_difference_images or write_debug_images:
            prefix = f"d_{iteration:02d}_"
            write_as_fits(f"{output_dir}/{prefix}{os.path.basename(im.f_name)}", im.difference,
                          supplementary_data={"INV_VAR": im.inv_var},
                          supplementary_header={"DX_INT": im.dx_int, "DY_INT": im.dy_int,
                                                "DX_SUB": im.dx_subpix, "DY_SUB": im.dy_subpix})
            prefix = f"vm_{iteration:02d}_"
            write_as_fits(f"{output_dir}/{prefix}{os.path.basename(im.f_name)}", im.vmask)
            prefix = f"e_{iteration:02d}_"
            r = im.mask * r / im.sigma
            r[np.isnan(r)] = 0.0
            write_as_fits(f"{output_dir}/{prefix}{os.path.basename(im.f_name)}", r)

        if iteration == 0:
            prefix = f"a_{iteration:02d}_"
            write_as_fits(f"{output_dir}/{prefix}{os.path.basename(im.f_name)}", im.data)
            prefix = f"m_{iteration:02d}_"
            write_as_fits(f"{output_dir}/{prefix}{os.path.basename(im.f_name)}", im.mask)

        if write_debug_images:
            prefix = f"z_{iteration:02d}_"
            write_as_fits(f"{output_dir}/{prefix}{os.path.basename(im.f_name)}", im.model)

            prefix = f"r_{iteration:02d}_"
            write_as_fits(f"{output_dir}/{prefix}{os.path.basename(im.f_name)}", r)

            prefix = f"m_{iteration:02d}_"
            write_as_fits(f"{output_dir}/{prefix}{os.path.basename(im.f_name)}", im.mask)



def mask_difference_image_residuals(images: list[Image], threshold: float = 3.0, iteration: int = 0,
                                    mask_only_positive: bool = True, output_dir: str = "Results") -> int:
    """Create a mask to flag high difference image residuals."""

    #gain = 20.0  # e-/ADU
    #RON = 0.128  # ADU
    #RON = np.median(np.abs(images[0].difference))
    #print(f"RON: {RON}")

    new_pixels_masked = 0

    for im in images:

        if im.vmask is None:
            last_vmask = np.ones_like(im.data, dtype=bool)
        else:
            last_vmask = im.vmask

        im.vmask = np.ones_like(im.data, dtype=bool)

        #im.noise = np.sqrt(im.model / gain + RON ** 2)
        #im.noise = np.sqrt(im.model / gain + RON ** 2)

        if mask_only_positive:
            im.vmask[im.difference > threshold * im.sigma] = 0
        else:
            im.vmask[im.difference ** 2 > (threshold * im.sigma) ** 2] = 0

        im.vmask_delta = np.logical_xor(im.vmask, last_vmask)

        #prefix = f"no_{iteration:02d}_"
        #write_as_fits(f"{output_dir}/{prefix}{os.path.basename(im.f_name)}", im.noise)

        new_pixels_masked += np.sum(im.vmask_delta)

    print(f"{new_pixels_masked} new pixels masked or unmasked for iteration {iteration:02d}.")

    return new_pixels_masked


def refine_offsets(images: list[Image]) -> np.ndarray:
    """Computes changes to the offsets defined for each image from the astrometric reference."""

    delta_xy = np.zeros((len(images), 2))

    A = np.zeros((2, 2))
    b = np.zeros(2)

    for k, im in enumerate(images):

        # im.inv_var is already set to zero for saturated pixels.
        # For this function we don't want to count adjacent pixels.

        iv = minimum_filter(im.inv_var, size=5, mode="constant", cval=0.0)

        A[0, 0] = np.sum(im.dRdx ** 2 * iv)
        A[0, 1] = np.sum(im.dRdx * im.dRdy * iv)
        A[1, 1] = np.sum(im.dRdy ** 2 * iv)
        A[1, 0] = A[0, 1]

        b[0] = -np.sum(im.difference * im.dRdx * iv)
        b[1] = -np.sum(im.difference * im.dRdy * iv)

        delta_xy[k, :] = np.linalg.solve(A, b)
        im.dx_subpix -= delta_xy[k, 0]
        im.dy_subpix -= delta_xy[k, 1]

        if im.dx_subpix < -0.5:
            im.dx_int -= 1
            im.dx_subpix += 1.0
        if im.dx_subpix > 0.5:
            im.dx_int += 1
            im.dx_subpix -= 1.0
        if im.dy_subpix < -0.5:
            im.dy_int -= 1
            im.dy_subpix += 1.0
        if im.dy_subpix > 0.5:
            im.dy_int += 1
            im.dy_subpix -= 1.0

        # This is a flag that forces a new oversampled construction for the whole image
        im.vmask_delta = np.zeros_like(im.data, dtype=bool)

    return delta_xy


def plot_offsets(offsets: np.ndarray, output_dir: str) -> None:
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

    plt.savefig(f"{output_dir}/offsets.png")


if __name__ == '__main__':
    pass
