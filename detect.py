#
#  Dazzle photometry module
#
import numpy as np
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture
from astropy.table import QTable
from astropy.io import fits
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize

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


def plot_magnitude_histogram(stars: QTable, file: str = "magnitudes.png") -> None:
    """Plot a histogram of the magnitudes of stars in the provided table."""

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.hist(stars['mag'], bins=21, range=(-10, 0))
    plt.savefig(file, bbox_inches="tight")


def display_detected_stars(im: np.ndarray, sources: QTable, file: str = 'detected_stars.png') -> None:
    """Make a plot of the image with detected stars indicated."""

    # Display the image with a "square root stretch" - this makes fainter stars show up better
    z_min = np.percentile(im, 2)
    z_max = np.percentile(im, 98)
    norm = ImageNormalize(stretch=SqrtStretch(), vmin=z_min, vmax=z_max)

    fig, ax = plt.subplots(1, 1, figsize=(30, 30))
    ax.imshow(im, cmap='Greys', origin='lower', norm=norm)

    # Plot circles (apertures) on top of the image at the locations
    # of the detected stars. Note that 'xcentroid' and 'ycentroid' are columns
    # in our sources table.
    positions = np.transpose((sources["xcentroid"], sources["ycentroid"]))
    apertures = CircularAperture(positions, r=20.0)
    apertures.plot(color='blue', lw=1.0, alpha=0.5)

    plt.savefig(file, bbox_inches="tight")


if __name__ == "__main__":
    with fits.open("oversampled.fits") as f:
        im = f[0].data

    stars_table = detect_stars(im)
    print(stars_table)

    plot_magnitude_histogram(stars_table)
    display_detected_stars(im, stars_table)
