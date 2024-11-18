import sys
import os
import numpy as np

import matplotlib.pyplot as plt

from astropy.io import fits
from photutils.psf import GriddedPSFModel

from scipy.optimize import curve_fit

from multiprocessing import Pool
from functools import partial

from context import utils, photometry

# For parallel processing. Set this to 1 if you don't want parallel processing.
# max_parallel_processes = 1
MAX_PARALLEL_PROCESSES = int(os.cpu_count() / 2)


def ulens_model1(t: np.ndarray, u0: float, t0: float, tE: float, base_flux: float) -> np.ndarray:
    """Return PSPL microlensing model for given parameters at t."""
    u = np.sqrt(u0 ** 2 + (t - t0) ** 2 / tE ** 2)
    a = (u ** 2 + 2) / (u * np.sqrt(u ** 2 + 4))
    return base_flux * (a - 1.0)


def ulens_model2(t: np.ndarray, u0: float, t0: float, tE: float, base_flux: float, base_measure: float) -> np.ndarray:
    """Return PSPL microlensing model for given parameters at t."""
    return ulens_model1(t, u0, t0, tE, base_flux) + base_measure


def fit_ulens(t: np.ndarray, f: np.ndarray, err_f: np.ndarray, params_guess: list[float] = None) -> list[float]:
    """Fit ulens model to given data."""
    print(f)
    good = ~np.isnan(f)
    if params_guess is None:
        params_guess = [0.01, 2180, 5.0, np.max(f[good])]
    params, cov = curve_fit(ulens_model1, t[good], f[good], p0=params_guess, sigma=err_f[good], absolute_sigma=True)
    params = params.tolist()
    params.append(0.0)
    params, cov = curve_fit(ulens_model2, t[good], f[good], p0=params, sigma=err_f[good], absolute_sigma=True)

    return params, cov


def test_single():
    #
    # Test of fitting PSF to single image
    #

    print("Single image test ...")

    with fits.open(f"{config_data['output_dir']}/d_02_synthpop_test18_t0144.fits") as f:
        d_im = f[0].data.T
        d_im_inv_var = f[1].data.T
        hdr = f[0].header
        dx_int = hdr["DX_INT"]
        dy_int = hdr["DY_INT"]
        dx_sub = hdr["DX_SUB"]
        dy_sub = hdr["DY_SUB"]

    x0, y0 = 459, 660
    rad = 5

    x0 += dx_int
    y0 += dy_int

    x = np.arange(x0 - rad, x0 + rad + 1)
    y = np.arange(y0 - rad, y0 + rad + 1)

    xx, yy = np.meshgrid(x, y)

    print("data grid from", x[0], y[0])
    pos = photometry.optimize_position(d_im[x[0]:x[-1] + 1, y[0]:y[-1] + 1],
                                       d_im_inv_var[x[0]:x[-1] + 1, y[0]:y[-1] + 1], psf_grid, xx, yy,
                                       (x0 + dx_sub, y0 + dy_sub))

    print("pos - (dx_int, dy_int)", pos[0] - dx_int, pos[1] - dy_int)

    z = psf_grid.evaluate(yy, xx, 1.0, pos[1], pos[0])
    z /= np.sum(z)

    flux, err_flux = photometry.fit_psf(z, d_im[x[0]:x[-1] + 1, y[0]:y[-1] + 1],
                                        d_im_inv_var[x[0]:x[-1] + 1, y[0]:y[-1] + 1])

    print(f"Flux = {flux} +/- {err_flux}.")

    fig, ax = plt.subplots(1, 3, figsize=(20, 6))

    d_min = np.min(d_im[x[0]:x[-1] + 1, y[0]:y[-1] + 1])
    d_max = np.max(d_im[x[0]:x[-1] + 1, y[0]:y[-1] + 1])

    cm1 = ax[0].imshow(z, origin='lower', extent=[y[0] - 0.5, y[-1] + 0.5, x[0] - 0.5, x[-1] + 0.5])
    cm2 = ax[1].imshow(d_im[x[0]:x[-1] + 1, y[0]:y[-1] + 1], origin='lower', vmin=-50, vmax=50,
                       extent=[y[0] - 0.5, y[-1] + 0.5, x[0] - 0.5, x[-1] + 0.5])
    _ = ax[2].imshow(d_im[x[0]:x[-1] + 1, y[0]:y[-1] + 1] - flux * z, origin='lower', vmin=-50, vmax=50,
                     extent=[y[0] - 0.5, y[-1] + 0.5, x[0] - 0.5, x[-1] + 0.5])

    ax[0].title.set_text('PSF')
    ax[1].title.set_text('Difference Image')
    ax[2].title.set_text('Residual')

    plt.colorbar(cm1, ax=ax, location="left")
    plt.colorbar(cm2, ax=ax, location="right")

    plt.savefig(f"{config_data['output_dir']}/phot_test_single.png")


def phot_multiple(images: list[photometry.Image], positions: np.ndarray, psf_grid: GriddedPSFModel,
                  image_offset: (int, int) = (0, 0),
                  root_out: str = "var",
                  position_convergence_size: int = 6, aperture_radius: int = 5,
                  make_image_plots: bool = False, plot_aperture_photometry: bool = False,
                  plot_ulens_model: bool = False, make_stamps: bool = False, direct_images: list[np.ndarray] = None):
    """Fit PSFs to image stack at given positions."""

    print("Multiple image photometry ...")

    for m in range(len(positions)):

        if not os.path.exists(f"{root_out}_{m:04d}_lightcurve.png"):

            # These must be int not float
            initial_ref_pos = (int(positions[m, 0]), int(positions[m, 1]))

            if positions[m, 0] < 10 or positions[m, 1] < 10 or positions[m, 0] > images[0].data.shape[0] - 10 or \
                    positions[m, 1] > images[0].data.shape[1] - 10:
                continue

            ref_pos = photometry.optimize_position_stack(images, initial_ref_pos, psf_grid, position_convergence_size,
                                                         image_offset=image_offset)

            x0 = ref_pos[0]
            y0 = ref_pos[1]

            flux = np.zeros(len(images))
            err_flux = np.zeros(len(images))
            aperture_flux = np.zeros(len(images))
            err_aperture_flux = np.zeros(len(images))

            columns_per_page = 2
            rows_per_page = 10
            images_per_page = columns_per_page * rows_per_page

            if direct_images is None:
                direct_images = ["" for im in images]

            for i, ims in enumerate(zip(images, direct_images)):

                im, direct_im = ims

                if make_image_plots:

                    counter = i % images_per_page
                    page = i // images_per_page

                    if counter == 0:
                        fig, ax = plt.subplots(rows_per_page, 3 * columns_per_page, figsize=(20, 30))

                    row = counter % rows_per_page
                    col = counter // rows_per_page

                xpos = x0 + im.dx_int + im.dx_sub
                ypos = y0 + im.dy_int + im.dy_sub

                xgrid = np.rint(xpos).astype(int)
                ygrid = np.rint(ypos).astype(int)

                x = np.arange(xgrid - aperture_radius, xgrid + aperture_radius + 1)
                y = np.arange(ygrid - aperture_radius, ygrid + aperture_radius + 1)
                xx, yy = np.meshgrid(x + image_offset[0], y + image_offset[1])

                pos = (xpos, ypos)

                z = psf_grid.evaluate(yy, xx, 1.0, pos[1] + image_offset[1], pos[0] + image_offset[0]).T

                print("xpos, ypos, image_offset, xx, yy, z", xpos, ypos, image_offset, xx, yy, z)

                z[(xx - xpos - image_offset[0]) ** 2 + (yy - ypos - image_offset[1]) ** 2 > aperture_radius ** 2] = 0.0

                z /= np.sum(z)

                flux[i], err_flux[i] = photometry.fit_psf(z, im.data[x[0]:x[-1] + 1, y[0]:y[-1] + 1],
                                                          im.inv_var[x[0]:x[-1] + 1, y[0]:y[-1] + 1])
                if im.inv_var[xgrid, ygrid] < 1e-6:
                    flux[i] = np.nan

                if plot_aperture_photometry:
                    aperture_flux[i] = np.sum(im.data[x[0]:x[-1] + 1, y[0]:y[-1] + 1])
                    err_aperture_flux[i] = np.sqrt(aperture_flux[i])

                if make_image_plots:

                    d_max = 0.5 * np.max([np.max(im.data[x[0]:x[-1] + 1, y[0]:y[-1] + 1]),
                                          -np.min(im.data[x[0]:x[-1] + 1, y[0]:y[-1] + 1])])

                    _ = ax[row, 3 * col + 0].imshow(z, origin='lower',
                                                    extent=[y[0] - 0.5, y[-1] + 0.5, x[0] - 0.5, x[-1] + 0.5])
                    cm2 = ax[row, 3 * col + 1].imshow(im.data[x[0]:x[-1] + 1, y[0]:y[-1] + 1], origin='lower',
                                                      vmin=-d_max, vmax=d_max,
                                                      extent=[y[0] - 0.5, y[-1] + 0.5, x[0] - 0.5, x[-1] + 0.5])
                    _ = ax[row, 3 * col + 2].imshow(im.data[x[0]:x[-1] + 1, y[0]:y[-1] + 1] - flux[i] * z,
                                                    origin='lower', vmin=-d_max,
                                                    vmax=d_max,
                                                    extent=[y[0] - 0.5, y[-1] + 0.5, x[0] - 0.5, x[-1] + 0.5])

                    fig.colorbar(cm2, ax=ax[row, 3 * col:3 * col + 3], shrink=0.9, location="right")

                    ax[row, 3 * col + 1].axes.xaxis.set_ticks([])
                    ax[row, 3 * col + 1].axes.yaxis.set_ticks([])
                    ax[row, 3 * col + 2].axes.xaxis.set_ticks([])
                    ax[row, 3 * col + 2].axes.yaxis.set_ticks([])

                    ax[row, 3 * col + 1].set_title(f"t = {15 * i}")
                    ax[row, 3 * col + 0].set_title(f"image {im.f_name.split('.')[-2][-4:]}")

                    if counter == columns_per_page * rows_per_page - 1:
                        plt.savefig(f"{root_out}_{m:04d}_{page:04d}.png")

                if make_stamps:
                    utils.write_as_fits(f"{root_out}_{m:04d}_{i:04d}_stamp_diff.fits",
                                        im.data[x[0]:x[-1] + 1, y[0]:y[-1] + 1])
                    utils.write_as_fits(f"{root_out}_{m:04d}_{i:04d}_stamp_pdiff.fits",
                                        im.data[x[0]:x[-1] + 1, y[0]:y[-1] + 1] - flux[i] * z)
                    try:
                        utils.write_as_fits(f"{root_out}_{m:04d}_{i:04d}_stamp_im.fits",
                                            direct_im[x[0]:x[-1] + 1, y[0]:y[-1] + 1])
                    except:
                        raise

            t = np.arange(len(images)) * 15

            np.savetxt(f"{root_out}_{m:04d}.phot",
                       np.vstack((t, flux, err_flux, aperture_flux, err_aperture_flux)).T,
                       fmt='%10.4f   %10.4f   %10.4f   %10.4f   %10.4f')

            fig, ax = plt.subplots(2, 1, figsize=(9, 8))

            ax[0].errorbar(t, flux, err_flux, fmt=".", c='b')

            imax = np.argmax(flux)
            ind_range = np.arange(np.max([0, imax - 10]), np.min([len(images), imax + 11]))

            if plot_aperture_photometry:
                ax[0].errorbar(t, aperture_flux, err_aperture_flux, fmt=".", c='c')

            if plot_ulens_model:

                try:
                    params, cov = fit_ulens(t, flux, err_flux)
                except RuntimeError:
                    params = None

                if params is not None:
                    tt = np.linspace(t[0], t[-1], 1001)
                    flux_model = ulens_model2(tt, *params)
                    ax[0].plot(tt, flux_model, 'r')
                    tt_zoom = np.linspace(t[ind_range][0], t[ind_range][-1], 1001)
                    flux_model_zoom = ulens_model2(tt_zoom, *params)
                    ax[1].plot(tt_zoom, flux_model_zoom, 'r')

            ax[0].tick_params(axis='y', which='both', direction='in', right=True)
            ax[0].tick_params(axis='x', which='both', direction='in', top=True)

            ax[0].set_xlabel('Time (min)')
            ax[0].set_ylabel('Flux')

            ax[1].errorbar(t[ind_range], flux[ind_range], err_flux[ind_range], fmt=".", c="b")

            if plot_aperture_photometry:
                ax[1].errorbar(t[ind_range], aperture_flux[ind_range], err_aperture_flux[ind_range], fmt=".", c="c")

            ax[1].tick_params(axis='y', which='both', direction='in', right=True)
            ax[1].tick_params(axis='x', which='both', direction='in', top=True)
            ax[1].set_xlabel('Time (min)')
            ax[1].set_ylabel('Flux')

            fig.tight_layout()
            plt.savefig(f"{root_out}_{m:04d}_lightcurve.png")


def process_image_section(config_data: dict, label: str, image_offset: (int, int) = (0, 0), timescales=None) -> None:

    if timescales is None:
        timescales = [2, 4]

    file_iteration_number = f"{config_data['difference_image_iterations']:02d}"

    output_dir = f"{config_data['output_dir']}{label}/detected_variables"
    files = [f"{output_dir}/{f}" for f in os.listdir(output_dir) if
             f"d_{file_iteration_number}_{config_data['data_root']}" in f and f.endswith(".fits")]
    files.sort()

    direct_files = [f.replace(f"d_{file_iteration_number}", f"a_{file_iteration_number}") for f in files]

    images = [photometry.Image(f) for f in files]

    direct_images = []
    for f_name in direct_files:
        with fits.open(f_name) as f:
            direct_images.append(f[0].data.T)

    for timescale in timescales:

        positions = np.loadtxt(f"{output_dir}/variables_{timescale}.txt")

        phot_multiple(images, positions[:, 1:3], psf_grid, image_offset,
                      f"{output_dir}/variables_{timescale}_d{file_iteration_number}",
                      position_convergence_size=4, aperture_radius=3, plot_ulens_model=True, make_image_plots=False,
                      make_stamps=False, direct_images=direct_images)


if __name__ == "__main__":

    #
    #  Configuration
    #

    print(sys.argv)

    config_data = utils.read_config(f"{os.path.dirname(__file__)}/{sys.argv[1]}")

    #
    #  Check for required config fields
    #

    required_fields = ["data_dir", "data_root", "output_dir", "input_xrange", "input_yrange"]
    for f in required_fields:
        if f not in config_data.keys():
            raise ValueError(f"Missing field {f} in configuration file {sys.argv[1]}.")

    if "expand_factor" not in config_data.keys():
        config_data["expand_factor"] = 1.2

    if "image_splits" not in config_data.keys():
        config_data["image_splits"] = 1

    if "difference_image_iterations" not in config_data.keys():
        config_data["difference_image_iterations"] = 5

    file_iteration_number = f"{config_data['difference_image_iterations']:02d}"

    PSF_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../PSFs'))
    PSF_file = "wfi_sca01_f146_fovp101_samp10_npsf9.fits"

    psf_grid = photometry.read_psf_grid(f"{PSF_dir}/{PSF_file}")

    #
    #   Set up image sections
    #

    section_size_x = (config_data["input_xrange"][1] - config_data["input_xrange"][0]) // config_data["image_splits"]
    section_size_y = (config_data["input_yrange"][1] - config_data["input_yrange"][0]) // config_data["image_splits"]

    x_ranges = []
    y_ranges = []
    labels = []
    for i in range(config_data["image_splits"]):
        for j in range(config_data["image_splits"]):
            x_ranges.append((config_data["input_xrange"][0] + i * section_size_x,
                             config_data["input_xrange"][0] + (i + 1) * section_size_x))
            y_ranges.append((config_data["input_yrange"][0] + j * section_size_y,
                             config_data["input_yrange"][0] + (j + 1) * section_size_y))
            labels.append(f"_{i}_{j}")

    #
    #   Process each image section
    #

    if MAX_PARALLEL_PROCESSES > 1:

        offsets = [(x[0], y[0]) for x, y in zip(x_ranges, y_ranges)]

        print('labels', labels)
        print('offsets', offsets)

        with Pool(np.min([config_data["image_splits"] ** 2, MAX_PARALLEL_PROCESSES])) as pool:
            pool.starmap(partial(process_image_section, config_data),
                         zip(labels, offsets))

    else:

        for x_range, y_range, label in zip(x_ranges, y_ranges, labels):
            process_image_section(config_data, label, image_offset=(x_range[0], y_range[0]))

