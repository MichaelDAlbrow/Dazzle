import sys
import os
import numpy as np
from multiprocessing import Pool
from functools import partial

#
# Uncomment for code profiling, and set MAX_PARALLEL_PROCESSES to 1
#
#import cProfile
#import pstats
#from pstats import SortKey
#import io

from context import dazzle, utils

# For parallel processing. Set this to 1 if you don't want parallel processing.
#MAX_PARALLEL_PROCESSES = 1
MAX_PARALLEL_PROCESSES = int(os.cpu_count() / 2)


def reduce_image_section(files: list[str], config_data: dict, input_xrange: tuple[int, int],
                         input_yrange: tuple[int, int], label: str) -> None:
    output_dir = f"{config_data['output_dir']}{label}"

    os.mkdir(output_dir)

    n_input_images = len(files)

    if "reference_stack_range" in config_data:
        reference_image_range = config_data["reference_stack_range"]
    else:
        reference_image_range = (0, n_input_images)

    images = [dazzle.Image(f, input_xrange, input_yrange) for f in files[:n_input_images]]

    #
    # Compute the offset in pixels between each image and the first one.
    #

    offsets = np.zeros((len(images), 2))
    for k, im in enumerate(images):
        offsets[k, :] = im.compute_offset(images[0])

    dazzle.plot_offsets(offsets, output_dir=output_dir)

    output_xrange = (-np.rint(np.min(offsets[:, 0])).astype(int),
                     images[0].data.shape[0] - np.rint(np.max(offsets[:, 0])).astype(int) - 1)
    output_yrange = (-np.rint(np.min(offsets[:, 1])).astype(int),
                     images[0].data.shape[1] - np.rint(np.max(offsets[:, 1])).astype(int) - 1)

    expand_factor = config_data["expand_factor"]

    theta = None

    n_masked_pixels = 1

    for iteration in range(config_data["difference_image_iterations"] + 1):

        print(f"Iteration {iteration + 1} for {label}:")

        if n_masked_pixels > 0:

            # Compute the coefficients of the basis functions for each image pixel
            print(f"Computing coefficients for {label}")

            theta = dazzle.solve_linear(images, output_xrange, output_yrange,
                                        reference_image_range=reference_image_range, save=False,
                                        output_dir=output_dir, expand_factor=expand_factor, theta=theta)

            # Compute and save an oversampled image
            print(f"Computing oversampled image for {label}")
            z = dazzle.evaluate_bipolynomial_basis(theta, output_xrange, output_yrange, expand_factor=expand_factor)

            print(f"Writing oversampled image for {label}")
            write_difference_images = False
            if iteration == int(config_data["difference_image_iterations"]):
                dazzle.write_as_fits(f"{output_dir}/oversampled_{iteration:02d}.fits", z)
                write_difference_images = True

            # Difference images
            print(f"Making difference images for {label}")
            dazzle.make_difference_images(images, theta, output_xrange, output_yrange, output_dir=output_dir,
                                          iteration=iteration, expand_factor=expand_factor, write_debug_images=False,
                                          write_difference_images=write_difference_images)

            # Mask high residual pixels
            print(f"Masking high residual pixels for {label}")
            if iteration % 4 != 0:
                mask_threshold = 4
                n_masked_pixels = dazzle.mask_difference_image_residuals(images, threshold=mask_threshold,
                                                                         output_dir=output_dir, iteration=iteration)

            # Refine offsets
            #if iteration % 4 == 0:
            #    _ = dazzle.refine_offsets(images)


if __name__ == "__main__":

    #
    # Uncomment for code profiling
    #
    #pr = cProfile.Profile()
    #pr.enable()

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

    #
    #  Set up data
    #

    files = [f"{config_data['data_dir']}/{f}" for f in os.listdir(config_data["data_dir"]) if
             config_data["data_root"] in f and f.endswith(".fits")]

    files.sort()

    print("Found files:")
    print(files)

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

        with Pool(MAX_PARALLEL_PROCESSES) as pool:
            pool.starmap(partial(reduce_image_section, files, config_data),
                         zip(x_ranges, y_ranges, labels))

    else:

        for x_range, y_range, label in zip(x_ranges, y_ranges, labels):
            reduce_image_section(files, config_data, x_range, y_range, label)

    #
    # Uncomment for code profiling
    #
    # pr.disable()
    #
    # s = io.StringIO()
    # sortby = SortKey.CUMULATIVE
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print(s.getvalue())
