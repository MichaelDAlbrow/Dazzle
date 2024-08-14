import sys
import os
import time
from context import photometry, utils

PSF_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../PSFs'))
PSF_file = "wfi_sca01_f146_fovp101_samp10_npsf9.fits"


start = time.perf_counter()

config_data = utils.read_config(f"{os.path.dirname(__file__)}/{sys.argv[1]}")


files = [f"{config_data['output_dir']}/{f}" for f in os.listdir(config_data["output_dir"]) if
         f"d_02_{config_data['data_root']}" in f and f.endswith(".fits")]
files.sort()

images = [photometry.Image(f) for f in files]

psf_grid = photometry.read_psf_grid(f"{PSF_dir}/{PSF_file}")

print(psf_grid)

end = time.perf_counter()
print(f"Elapsed time: {end - start:0.2f} seconds")

