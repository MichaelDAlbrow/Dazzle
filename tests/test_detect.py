import sys
import os

import numpy as np

import matplotlib.pyplot as plt

from astropy.io import fits
from scipy.stats import median_abs_deviation

from context import photometry, utils, detect


def save_variables(variables: dict, file_root: str):
    """Save variable star data to file."""

    for timescale, vars in variables.items():

        if not os.path.exists(f"{file_root}_{timescale}"):

            np.savetxt(f"{file_root}_{timescale}.txt", vars, fmt='%5d')


def plot_variables(images: list[photometry.Image], variables: dict, plot_dir: str = None, display_image: str = None,
                   make_image_plots: bool = False):
    """Plot detected transient variables."""

    for timescale, vars in variables.items():

        if display_image is not None:

            with fits.open(display_image) as f:
                d_im = f[0].data.T

            if vars.shape[0] > 0:
                pos = np.zeros((vars.shape[0], 2))
                vars = np.atleast_2d(vars)
                pos[:, 0] = vars[:, 2]
                pos[:, 1] = vars[:, 1]
                detect.display_detected_stars(d_im, positions=pos, file=f"{plot_dir}/variables_{timescale}.png")

        if make_image_plots:

            fig, ax = plt.subplots(len(variables), 3, figsize=(20, len(vars) * 6 + 1))

            for i, var in enumerate(vars):

                sub_image_1 = images[var[0] - 1].data[
                              var[1] + images[var[0] - 1].dx_int - 10:var[1] + images[var[0] - 1].dx_int + 11,
                              var[2] + images[var[0] - 1].dy_int - 10:var[2] + images[var[0] - 1].dy_int + 11]
                sub_image_2 = images[var[0]].data[
                              var[1] + images[var[0]].dx_int - 10:var[1] + images[var[0]].dx_int + 11,
                              var[2] + images[var[0]].dy_int - 10:var[2] + images[var[0]].dy_int + 11]
                sub_image_3 = images[var[0] + 1].data[
                              var[1] + images[var[0] + 1].dx_int - 10:var[1] + images[var[0] + 1].dx_int + 11,
                              var[2] + images[var[0] + 1].dy_int - 10:var[2] + images[var[0] + 1].dy_int + 11]

                d_max = 10.0 * median_abs_deviation(images[var[0]].data, axis=None)

                ax[i, 0].imshow(sub_image_1, vmin=-d_max, vmax=d_max, origin='lower')
                ax[i, 1].imshow(sub_image_2, vmin=-d_max, vmax=d_max, origin='lower')
                ax[i, 2].imshow(sub_image_3, vmin=-d_max, vmax=d_max, origin='lower')
                ax[i, 0].title.set_text(f"({var[1]},{var[2]})")
                ax[i, 0].set_xlabel(f"{var[0] - 1}")
                ax[i, 1].set_xlabel(f"{var[0]}")
                ax[i, 2].set_xlabel(f"{var[0] + 1}")

            plt.savefig(f"{plot_dir}/variables_images_{timescale}.png")


if __name__ == "__main__":

    config_data = utils.read_config(f"{os.path.dirname(__file__)}/{sys.argv[1]}")

    output_dir = f"{config_data['output_dir']}/detected_variables"

    files = [f"{config_data['output_dir']}/{f}" for f in os.listdir(config_data["output_dir"]) if
             f"d_05_{config_data['data_root']}" in f and f.endswith(".fits")]
    files.sort()

    images = [photometry.Image(f) for f in files]

    if os.path.exists(output_dir):

        variables = {}
        variable_files = [f for f in os.listdir(output_dir) if "variables_" in f and f.endswith(".txt")]
        for f in variable_files:
            temporal_sigma = f.split("_")[-1].split(".")[0]
            variables[f"{temporal_sigma}"] = np.loadtxt(f"{output_dir}/{f}")

    else:

        os.mkdir(output_dir)
        variables = detect.detect_variables_from_difference_image_stack(images, threshold=4)
        save_variables(variables, f"{output_dir}/variables")

    plot_variables(images, variables, plot_dir=output_dir, display_image=images[144].f_name)

