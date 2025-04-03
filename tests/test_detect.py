import sys
import os

import numpy as np

import matplotlib.pyplot as plt

from astropy.io import fits
from scipy.stats import median_abs_deviation

from multiprocessing import Pool
from functools import partial

from context import utils, photometry, detect

# For parallel processing. Set this to 1 if you don't want parallel processing.
#MAX_PARALLEL_PROCESSES = 1
MAX_PARALLEL_PROCESSES = int(os.cpu_count()/2)


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


def process_image_section(config_data: dict, label: str) -> None:

    file_iteration_number = f"{config_data['difference_image_iterations']:02d}"

    output_dir = f"{config_data['output_dir']}{label}"
    files = [f"{output_dir}/{f}" for f in os.listdir(output_dir) if
            f"d_{file_iteration_number}_{config_data['data_root']}" in f and f.endswith(".fits") and f[0].isalpha()]
    files.sort()

    images = [photometry.Image(f) for f in files]

    output_dir = f"{config_data['output_dir']}{label}/detected_variables"

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

    print(label, variables)

    plot_variables(images, variables, plot_dir=output_dir, display_image=images[144].f_name)


if __name__ == "__main__":

    config_data = utils.read_config(f"{os.path.dirname(__file__)}/{sys.argv[1]}")

    #
    #  Check for required config fields
    #

    required_fields = ["data_dir", "data_root", "output_dir", "input_xrange", "input_yrange"]
    for f in required_fields:
        if f not in config_data.keys():
            raise ValueError(f"Missing field {f} in configuration file {sys.argv[1]}.")

    if "image_splits" not in config_data.keys():
        config_data["image_splits"] = 1

    if "difference_image_iterations" not in config_data.keys():
        config_data["difference_image_iterations"] = 5

    file_iteration_number = f"{config_data['difference_image_iterations']:02d}"

    #
    #   Set up image sections
    #

    labels = []
    for i in range(config_data["image_splits"]):
        for j in range(config_data["image_splits"]):
            labels.append(f"_{i}_{j}")

    #
    #   Process each image section
    #

    if MAX_PARALLEL_PROCESSES > 1:

        with Pool(np.min([config_data["image_splits"]**2, MAX_PARALLEL_PROCESSES])) as pool:
            pool.map(partial(process_image_section, config_data), labels)

    else:

        for label in labels:

            process_image_section(config_data, label)

