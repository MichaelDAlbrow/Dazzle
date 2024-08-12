import os
import time
import numpy as np
import dazzle

start = time.perf_counter()

files = [f"Data/test18/{f}" for f in os.listdir("Data/test18") if "synthpop_test18_t" in f and f.endswith(".fits")]
files.sort()

# input_yrange = (2080, 2100)
# input_xrange = (2090, 2110)
input_yrange = (1800, 2200)
input_xrange = (1800, 2200)

n_input_images = len(files)
reference_image_range = (0, n_input_images)

images = [dazzle.Image(f, input_xrange, input_yrange) for f in files[:n_input_images]]

# Compute the offset in pixels between each image and the first one.

offsets = np.zeros((len(images), 2))
for k, im in enumerate(images):
    offsets[k, :] = im.compute_offset(images[0])

print("offsets standard deviation:", np.std(offsets[:, 0]), np.std(offsets[:, 1]))
dazzle.plot_offsets(offsets)

output_xrange = (-np.rint(np.min(offsets[:, 0])).astype(int),
                 images[0].data.shape[0] - np.rint(np.max(offsets[:, 0])).astype(int) - 1)
output_yrange = (-np.rint(np.min(offsets[:, 1])).astype(int),
                 images[0].data.shape[1] - np.rint(np.max(offsets[:, 1])).astype(int) - 1)

print("Input ranges:", input_xrange, input_yrange)
print("Output ranges:", output_xrange, output_yrange)

end = time.perf_counter()
print(f"Elapsed time: {end - start:0.2f} seconds")

# test refine offsets
# images[-1].dx_subpix += 0.05
# images[-1].dy_subpix += 0.1
# for im in images:
#     im.dx_subpix += 0.001 * np.random.randn()
#     im.dy_subpix += 0.001 * np.random.randn()

for iteration in range(3):

    print(f"Iteration {iteration+1}:")

    # Compute the coefficients of the basis functions for each image pixel
    print("Computing coefficients ...")
    theta = dazzle.solve_linear(images, output_xrange, output_yrange, reference_image_range=reference_image_range)
    end = time.perf_counter()
    print(f"Elapsed time: {end - start:0.2f} seconds")

    # Compute and save an oversampled image
    print("Computing oversampled image ...")
    z = dazzle.evaluate_bicubic_legendre(theta, output_xrange, output_yrange)
    end = time.perf_counter()
    print(f"Elapsed time: {end - start:0.2f} seconds")

    print("Writing oversampled image ...")
    dazzle.write_as_fits(f"Results/test18_oversampled_{iteration:02d}.fits", z)
    end = time.perf_counter()
    print(f"Elapsed time: {end - start:0.2f} seconds")

    # Difference images
    print("Making difference images  ...")
    dazzle.make_difference_images(images, theta, output_xrange, output_yrange, iteration=iteration)
    end = time.perf_counter()
    print(f"Elapsed time: {end - start:0.2f} seconds")

    # if iteration < 3:
    #
    #     # Refining offsets
    #     print("Refining offsets ...")
    #     dazzle.refine_offsets(images, output_xrange, output_yrange)
    #     end = time.perf_counter()
    #     print(f"Elapsed time: {end - start:0.2f} seconds")
    #
    # else:

    # Mask high residual pixels
    print("Masking high residual pixels ...")
    dazzle.mask_difference_image_residuals(images)
    end = time.perf_counter()
    print(f"Elapsed time: {end - start:0.2f} seconds")

