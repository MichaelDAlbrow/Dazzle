#
#  Dazzle detect module
#

import numpy as np
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture
from astropy.table import QTable
from astropy.io import fits
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize

from scipy.ndimage import convolve, maximum_filter, binary_dilation
from scipy.signal import correlate

from .utils import write_as_fits

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")


def detect_stars(im: np.ndarray | str, threshold: float = 100.0) -> QTable:
    """Detect stars on image im.
    We assume this is an oversampled image with PSF FWHM of ~ 10 pixels.
    A future improvement would be to use photutils.detection.StarFinder with a PSF from WebbPSF.
    """

    if isinstance(im, str):
        with fits.open(im) as f:
            im = f[0].data

    finder = DAOStarFinder(fwhm=10.0, threshold=threshold)
    sources = finder(im - np.min(im))

    return sources


def romanisim_artifacts_mask(im: np.ndarray, outfile=None) -> np.ndarray:
    """Detect and mask the square residuals left around saturated stars by RomainISIM/WebbPSF."""

    mask = np.ones_like(im)
    threshold = 100
    kernel = np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                       [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                       [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])
    c_im = convolve(im, kernel, mode='constant')
    mask[abs(c_im) > threshold] = 0
    c_im = convolve(im, kernel.T, mode='constant')
    mask[abs(c_im) > threshold] = 0
    mask = 1 - binary_dilation(1-mask).astype(mask.dtype)
    if outfile is not None:
        write_as_fits(outfile, mask)
    return mask


def detect_variables_from_difference_image_stack(images: list['Image'], threshold: float = 50.0, sigmas: list = [2, 4, 8, 16]) -> dict:
    """Detect variables from difference image stack."""

    #
    #  Stack all the difference images onto a common grid (i.e. remove the integer-pixel dithers).
    #

    dx_images = [im.dx_int for im in images]
    dy_images = [im.dy_int for im in images]
    dx_min = np.min(dx_images)
    dx_max = np.max(dx_images)
    dy_min = np.min(dy_images)
    dy_max = np.max(dy_images)

    shape = images[0].data.shape
    stack = np.zeros((len(images), shape[0]+dx_max-dx_min, shape[1]+dy_max-dy_min))

    mask_kernel = np.ones((5, 5))

    for i, im in enumerate(images):

        #
        # Mask near saturated pixels
        #
        mask = im.inv_var < 0.01
        mask = 1 - convolve(mask, mask_kernel, mode='constant')

        #
        # Also mask square artifacts left by RomanISIM
        #
        artifact_mask = romanisim_artifacts_mask(im.data * im.inv_var)

        x0 = -im.dx_int + dx_max
        y0 = -im.dy_int + dy_max

        stack[i, x0:x0+shape[0], y0:y0+shape[1]] = im.data * mask * artifact_mask

    #
    #  Filter the stack by convolving with 3d gaussians designed to approximately
    #  match the PSF size in the spatial directions, and various timescales in the temporal direction.
    #

    spatial_sigma = 1.0
    spatial_sigma_int = np.ceil(spatial_sigma).astype(int)

    edge_mask = np.ones_like(stack)
    edge_mask[:, :(dx_max+10), :] = 0
    edge_mask[:, (dx_min-10):, :] = 0
    edge_mask[:, :, :(dy_max+10)] = 0
    edge_mask[:, :, (dy_min-10):] = 0
    edge_mask[:2, :, :] = 0
    edge_mask[-2:, :, :] = 0

    peak_locations = {}

    for temporal_sigma in sigmas:

        temporal_sigma_int = np.ceil(temporal_sigma).astype(int)

        kernel_t_range = np.linspace(-temporal_sigma_int, temporal_sigma_int, 2*temporal_sigma_int+1)
        kernel_s_range = np.linspace(-spatial_sigma_int, spatial_sigma_int, 2*spatial_sigma_int+1)
        x, y, z = np.meshgrid(kernel_t_range, kernel_s_range, kernel_s_range, indexing='ij')
        kernel = np.exp(-(x**2/(2*temporal_sigma**2) + y**2/(2*spatial_sigma**2) + z**2/(2*spatial_sigma**2)))
        kernel /= np.sum(kernel)
        conv_stack = convolve(stack, kernel, mode='constant', cval=0.0)

        #
        #  Find peaks.
        #
        local_peaks = maximum_filter(conv_stack, size=(3*temporal_sigma_int+1, 3*spatial_sigma_int+1, 3*spatial_sigma_int+1)) == conv_stack
        threshold_cut = conv_stack > threshold
        significance_map = local_peaks * threshold_cut
        loc = np.where((significance_map == 1) & (edge_mask == 1))

        peak_locations[f"{temporal_sigma}"] = np.zeros((len(loc[0]), 4))

        peak_locations[f"{temporal_sigma}"][:, :3] = np.array(loc).T

        for m in range(len(loc[0])):

            i = peak_locations[f"{temporal_sigma}"][m, 0].astype(int)
            j = peak_locations[f"{temporal_sigma}"][m, 1].astype(int)
            k = peak_locations[f"{temporal_sigma}"][m, 2].astype(int)

            print(i, j, k, temporal_sigma, stack[i, j, k], conv_stack[i, j, k], edge_mask[i, j, k])

            peak_locations[f"{temporal_sigma}"][m, 3] = conv_stack[i, j, k]

    return peak_locations


def plot_magnitude_histogram(stars: QTable, file: str = "magnitudes.png") -> None:
    """Plot a histogram of the magnitudes of stars in the provided table."""

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.hist(stars['mag'], bins=21, range=(-10, 0))
    plt.savefig(file, bbox_inches="tight")


def display_detected_stars(im: np.ndarray, sources: QTable = None, positions: np.ndarray = None, file: str = 'detected_stars.png') -> None:
    """
    Make a plot of the image with detected stars indicated.
    Stars must be provided either in an astropy QTable or as a numpy array with 2 columns.
    """

    # Display the image with a "square root stretch" - this makes fainter stars show up better
    z_min = np.percentile(im, 2)
    z_max = np.percentile(im, 98)
    norm = ImageNormalize(stretch=SqrtStretch(), vmin=z_min, vmax=z_max)

    fig, ax = plt.subplots(1, 1, figsize=(30, 30))
    ax.imshow(im, cmap='Greys', origin='lower', norm=norm)

    # Plot circles (apertures) on top of the image at the locations
    # of the detected stars. Note that 'xcentroid' and 'ycentroid' are columns
    # in our sources table.
    if sources is QTable:
        positions = np.transpose((sources["xcentroid"], sources["ycentroid"]))

    if positions is not None:

        apertures = CircularAperture(positions, r=20.0)
        apertures.plot(color='blue', lw=1.0, alpha=0.5)
        plt.savefig(file, bbox_inches="tight")


if __name__ == "__main__":
    with fits.open("../oversampled.fits") as f:
        im = f[0].data

    stars_table = detect_stars(im)
    print(stars_table)

    plot_magnitude_histogram(stars_table)
    display_detected_stars(im, stars_table)
