import os
import json
import numpy as np
from astropy.io import fits

def read_config(config_file: str) -> dict:

    with open(config_file) as file:
        config_data = json.load(file)

    for field in ["data_dir", "data_root", "output_dir", "input_xrange", "input_yrange"]:
        if field not in config_data:
            raise Exception("Missing field {field} in {config_file}.")

    if not os.path.isdir(config_data["output_dir"]):
        os.mkdir(config_data["output_dir"])

    return config_data


def write_as_fits(f_name: str, data: np.ndarray, supplementary_data: dict = None, supplementary_header: dict = None,
                  overwrite: bool = True) -> None:
    """
    Write an image array to a FITS file.
    If provided, supplementary_data should be a dictionary of image arrays to be saved as extensions.
    """

    if not f_name.endswith(".fits"):
        f_name += ".fits"

    hdr = fits.Header()
    if supplementary_header is not None:
        for key, value in supplementary_header.items():
            hdr[key] = value

    data_HDU = fits.PrimaryHDU(data.T.astype(np.double), header=hdr)
    HDU_list = fits.HDUList([data_HDU])

    if supplementary_data is not None:
        for key, value in supplementary_data.items():
            HDU = fits.ImageHDU(data=value.T, name=key)
            HDU_list.append(HDU)

    HDU_list.writeto(f_name, overwrite=overwrite)

