import numpy as np
from astropy.io import fits
from photutils.psf import webbpsf_reader, GriddedPSFModel
from scipy.optimize import minimize

__author__ = "Michael Albrow"


class Image:

    def __init__(self, f_name: str) -> None:

        # We assume that difference images have been saved with an extension containing the inverse variance,
        # and header fields specifying the integer and subpixel offsets from a reference point.

        if f_name is None:
            raise ValueError("File name must be provided")

        self.f_name = f_name

        try:

            with fits.open(f_name) as f:

                self.f_name = f_name
                self.header = f[0].header
                self.data = f[0].data.T
                self.inv_var = f[1].data.T

                self.dx_sub = f[0].header['DX_SUB']
                self.dy_sub = f[0].header['DY_SUB']
                self.dx_int = f[0].header['DX_INT']
                self.dy_int = f[0].header['DY_INT']

        except OSError as e:

            print("OS error when reading file", f_name)
            raise e


def read_psf_grid(f_name: str) -> GriddedPSFModel:
    """Read PSF grid created by WebbPSF."""

    grid = webbpsf_reader(f_name)

    return grid


def fit_psf(psf: np.ndarray, im: np.ndarray, inv_var: np.ndarray) -> tuple[float, float]:
    """Optimal fit of psf to im at pos. """

    psf2 = np.sum(psf**2 * inv_var)
    flux = np.sum(psf*im*inv_var) / psf2
    err_flux = np.sqrt(np.sum(psf**2) / psf2)

    return flux, err_flux


def fit_psf_base(psf: np.ndarray, im: np.ndarray, inv_var: np.ndarray) -> (float, float, float):
    """Optimal fit of psf plus a constant to im at pos. """

    A = np.array([[1.0, np.sum(psf*inv_var)], [np.sum(psf*inv_var), np.sum(psf**2*inv_var)]])
    b = np.array([np.sum(im*inv_var), np.sum(psf*im*inv_var)])

    try:
        c = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan

    flux = c[1]
    err_flux = np.sqrt(np.sum(psf**2) / np.sum(psf**2*inv_var))

    return flux, err_flux, c[0]


def chi2_fit(pos: tuple[float, float], im: np.ndarray, inv_var: np.ndarray, psf_grid: GriddedPSFModel,
             xx: np.ndarray, yy: np.ndarray, image_offset: (int, int) = (0, 0), fit_radius: int = 2) -> float:
    """Evaluate PSF from grid at pos and return chi^2 for optimal fit to im."""

    z = psf_grid.evaluate(yy, xx, 1.0, pos[1]+image_offset[1], pos[0]+image_offset[0]).T

    z[(xx - pos[0] - image_offset[0]) ** 2 + (yy - pos[1] - image_offset[1]) ** 2 > fit_radius ** 2] = 0.0

    flux, err_flux = fit_psf(z, im, inv_var)
    #flux, err_flux, base = fit_psf_base(z, im, inv_var)

    return np.sum((im - flux*z)**2 * inv_var)


def chi2_fit_stack(ref_pos: tuple[float, float], images: list[Image], psf_grid: GriddedPSFModel, rad: int,
                   image_offset: (int, int) = (0, 0)) -> float:
    """Return total chi^2 of fitting PSF from grid at pos to images."""

    chi2 = 0.0

    x0 = ref_pos[0]
    y0 = ref_pos[1]

    for im in images:

        xpos = x0 + im.dx_int + im.dx_sub
        ypos = y0 + im.dy_int + im.dy_sub

        xgrid = np.rint(xpos).astype(int)
        ygrid = np.rint(ypos).astype(int)

        x = np.arange(xgrid - rad, xgrid + rad + 1)
        y = np.arange(ygrid - rad, ygrid + rad + 1)
        xx, yy = np.meshgrid(x+image_offset[0], y+image_offset[1])

        pos = (xpos, ypos)

        chi2 += chi2_fit(pos, im.data[x[0]:x[-1]+1, y[0]:y[-1]+1], im.inv_var[x[0]:x[-1]+1, y[0]:y[-1]+1],
                         psf_grid, xx, yy, image_offset)

    return chi2


def position_out_of_bounds(position: np.ndarray, radius: int, im_size: tuple[int, int], tolerance: int = 5) -> bool:
    """Return True if position is out of bounds."""

    test_radius = radius + tolerance
    if test_radius < position[0] < im_size[0] - test_radius and test_radius < position[1] < im_size[1] - test_radius:
        return False

    return True


def optimize_position(im: np.ndarray, inv_var: np.ndarray, psf_grid: GriddedPSFModel, xx: np.ndarray, yy: np.ndarray,
                      pos: tuple[float, float], image_offset: (int, int) = (0, 0)) -> tuple[float, float]:
    """Optimize star position for PSF fit to single image."""

    x0 = pos[0]
    y0 = pos[1]

    #
    # Initial grid search
    #
    chi2min = 1.e20
    for dx in np.linspace(-0.5, 0.5, 11):
        for dy in np.linspace(-0.5, 0.5, 11):
            pos = np.array([x0+dx, y0+dy])
            chi2 = chi2_fit(pos, im, inv_var, psf_grid, xx, yy, image_offset)
            if chi2 < chi2min:
                chi2min = chi2
                x1 = x0+dx
                y1 = y0+dy

    print(f"Grid min chi2 of {chi2min} at {x1}, {y1}.")

    initial_pos = np.array([x1, y1])

    result = minimize(chi2_fit, initial_pos, args=(im, inv_var, psf_grid, xx, yy, image_offset),
                      method='Nelder-Mead', tol=1.0e-8)

    print(result)

    return result.x


def optimize_position_stack(images: list[Image], ref_pos: tuple[int, int], psf_grid: GriddedPSFModel, rad: int = 5,
                            image_offset: (int, int) = (0, 0)) -> tuple[float, float]:

    #
    # Initial grid search
    #

    x0 = ref_pos[0]
    y0 = ref_pos[1]

    x1 = x0
    y1 = y0

    chi2_min = 1.e20

    for dx in np.linspace(-2, 2, 5):
        for dy in np.linspace(-2, 2, 5):

            chi2 = 0.0
            for im in images:

                xpos = x0 + im.dx_int + im.dx_sub + dx
                ypos = y0 + im.dy_int + im.dy_sub + dy

                xgrid = np.rint(xpos).astype(int)
                ygrid = np.rint(ypos).astype(int)

                x = np.arange(xgrid - rad, xgrid + rad + 1)
                y = np.arange(ygrid - rad, ygrid + rad + 1)
                xx, yy = np.meshgrid(x+image_offset[0], y+image_offset[1])

                pos = (xpos, ypos)

                if x[0] > 0 and y[0] > 0 and x[-1] < im.data.shape[0] - 1 and y[-1] < im.data.shape[1] - 1:

                    chi2 += chi2_fit(pos, im.data[x[0]:x[-1]+1, y[0]:y[-1]+1],
                                 im.inv_var[x[0]:x[-1]+1, y[0]:y[-1]+1],
                                 psf_grid, xx, yy, image_offset)

            if chi2 < chi2_min:
                chi2_min = chi2
                x1 = x0 + dx
                y1 = y0 + dy

    print(f"Grid min chi2 of {chi2_min} at ({x1}, {y1}).")

    initial_ref_pos = np.array([x1, y1])

    if position_out_of_bounds(initial_ref_pos, rad, images[0].data.shape):
        raise ValueError("Position out of bounds")

    initial_simplex = np.zeros((3, 2))
    initial_simplex[0, :] = initial_ref_pos
    initial_simplex[1, :] = initial_ref_pos + np.array([0.05, 0.05])
    initial_simplex[2, :] = initial_ref_pos + np.array([0.02, -0.02])

    result = minimize(chi2_fit_stack, initial_ref_pos, args=(images, psf_grid, rad, image_offset), method='Nelder-Mead',
                      options={'initial_simplex': initial_simplex})

    print(result)
    print(f"Converged min chi2 of {result.fun} at ({result.x[0]:.3f}, {result.x[-1]:.3f}).")

    return result.x


if __name__ == '__main__':

    pass
