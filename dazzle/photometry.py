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

        with fits.open(f_name) as f:

            self.f_name = f_name
            self.header = f[0].header
            self.data = f[0].data.T
            self.inv_var = f[1].data.T

            self.dx_sub = f[0].header['DX_SUB']
            self.dy_sub = f[0].header['DY_SUB']
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


def fit_psf(psf: np.ndarray, im: np.ndarray, inv_var: np.ndarray) -> tuple[float, float]:
    """Optimal fit of psf to im at pos. """

    flux = np.sum(psf*im*inv_var) / np.sum(psf**2 * inv_var)
    err_flux = np.sum(psf)**2 / np.sum(psf**2 * inv_var)

    return flux, err_flux


def chi2_fit(pos: tuple[float, float], im: np.ndarray, inv_var: np.ndarray, psf_grid: GriddedPSFModel,
             xx: np.ndarray, yy: np.ndarray) -> float:
    """Evaluate PSF from grid at pos and return chi^2 for optimal fit to im."""

    z = psf_grid.evaluate(yy, xx, 1.0, pos[1], pos[0])
    z /= np.sum(z)

    flux, err_flux = fit_psf(z, im, inv_var)

    #print("psf, im, inv_var ", z[0, 0], im[0, 0], inv_var[0, 0])

    return np.sum((im - flux*z)**2 * inv_var)


def chi2_fit_stack(ref_pos: tuple[float, float], images: list[Image], psf_grid: GriddedPSFModel, rad: int) -> float:
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
        xx, yy = np.meshgrid(x, y)

        pos = (xpos, ypos)

        z = psf_grid.evaluate(yy, xx, 1.0, pos[1], pos[0])
        z /= np.sum(z)

        flux, err_flux = fit_psf(z, im.data[x[0]:x[-1]+1, y[0]:y[-1]+1], im.inv_var[x[0]:x[-1]+1, y[0]:y[-1]+1])

        chi2 += np.sum((im.data[x[0]:x[-1]+1, y[0]:y[-1]+1] - flux*z)**2 * im.inv_var[x[0]:x[-1]+1, y[0]:y[-1]+1])

    return chi2


def optimize_position(im: np.ndarray, inv_var: np.ndarray, psf_grid: GriddedPSFModel, xx: np.ndarray, yy: np.ndarray, pos: tuple[float, float]) -> tuple[float, float]:
    """Optimize star position for PSF fit to single image."""

    x0 = pos[0]
    y0 = pos[1]

    #
    # Initial grid search
    #
    chi2min = 1.e6
    for dx in np.linspace(-0.5, 0.5, 11):
        for dy in np.linspace(-0.5, 0.5, 11):
            pos = np.array([x0+dx, y0+dy])
            chi2 = chi2_fit(pos, im, inv_var, psf_grid, xx, yy)
            print(dx, dy, pos, chi2, xx[0, 0], yy[0, 0], im[0, 0])
            if chi2 < chi2min:
                chi2min = chi2
                x1 = x0+dx
                y1 = y0+dy

    print(f"Grid min chi2 of {chi2min} at {x1}, {y1}.")

    initial_pos = np.array([x1, y1])

    result = minimize(chi2_fit, initial_pos, args=(im, inv_var, psf_grid, xx, yy), method='Nelder-Mead', tol=1.0e-8)

    print(result)

    return result.x


def optimize_position_stack(images: list[Image], ref_pos: tuple[int, int], psf_grid: GriddedPSFModel, rad: int = 5) -> tuple[float, float]:

    #
    # Initial grid search
    #

    x0 = ref_pos[0]
    y0 = ref_pos[1]

    chi2_min = 1.e6

    for dx in np.linspace(-2.5, 2.5, 51):
        for dy in np.linspace(-2.5, 2.5, 51):

            chi2 = 0.0
            for im in images:

                xpos = x0 + im.dx_int + im.dx_sub + dx
                ypos = y0 + im.dy_int + im.dy_sub + dy

                xgrid = np.rint(xpos).astype(int)
                ygrid = np.rint(ypos).astype(int)

                x = np.arange(xgrid - rad, xgrid + rad + 1)
                y = np.arange(ygrid - rad, ygrid + rad + 1)
                xx, yy = np.meshgrid(x, y)

                pos = (xpos, ypos)

                #print("data grid from", x[0], y[0])

                chi2 += chi2_fit(pos, im.data[x[0]:x[-1]+1, y[0]:y[-1]+1],
                                 im.inv_var[x[0]:x[-1]+1, y[0]:y[-1]+1],
                                 psf_grid, xx, yy)

            print(dx, dy, chi2)

            if chi2 < chi2_min:
                chi2_min = chi2
                x1 = x0 + dx
                y1 = y0 + dy

    print(f"Grid min chi2 of {chi2_min} at {x1}, {y1}.")

    initial_ref_pos = np.array([x1, y1])

    initial_simplex = np.zeros((3, 2))
    initial_simplex[0, :] = initial_ref_pos
    initial_simplex[1, :] = initial_ref_pos + np.array([0.05, 0.05])
    initial_simplex[2, :] = initial_ref_pos + np.array([0.02, -0.02])

    result = minimize(chi2_fit_stack, initial_ref_pos, args=(images, psf_grid, rad), method='Nelder-Mead', options={'initial_simplex': initial_simplex})

    print(result)

    return result.x


if __name__ == '__main__':

    pass
