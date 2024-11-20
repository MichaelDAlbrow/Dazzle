# Dazzle

A python package for over-sampled image construction, difference-imaging, transient detection, and difference-image PSF
photometry for the Nancy Grace Roman Space Telescope.


# Usage

The tests directory contains sample scripts for running the package. Typically they would be run in the order:

python test_difference_images.py config_paper_split.json

python test_detect.py config_paper_split.json

python test_photometry.py config_paper_split.json

The config file, and the scripts, should be edited for your specific purpose.

# Configuration

A sample json configuration file looks like:

{
  "data_dir": "/home/users/mda45/RomanISIM/Data/paper",
  "data_root": "dazzle_paper_t",
  "output_dir": "/home/users/mda45/RomanISIM/Data/paper_results",
  "input_xrange": [
    0,
    4088
  ],
  "input_yrange": [
    0,
    4088
  ],
  "image_splits": 4,
  "difference_image_iterations": 5
}

Most of this is self explanatory. Setting image_splits = 4 means 
that the 4088 x 4088 images are split into 16 x (1022 x 1022)
size, processed in parallel, and with the output sent to different 
directories.



