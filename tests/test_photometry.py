import sys
import os
import time
import numpy as np

import matplotlib.pyplot as plt
from scipy.ndimage import convolve

from astropy.io import fits

from context import photometry, utils

import webbpsf

PSF_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../PSFs'))
PSF_file = "wfi_sca01_f146_fovp101_samp10_npsf9.fits"

#
#
#
start = time.perf_counter()
print("Setting up ...")

config_data = utils.read_config(f"{os.path.dirname(__file__)}/{sys.argv[1]}")

psf_grid = photometry.read_psf_grid(f"{PSF_dir}/{PSF_file}")

#
# Blur the PSF grid to convert PSF to ePSF
#
kernel = np.ones((5, 5))
for j in range(psf_grid.data.shape[0]):
    frame = psf_grid.data[j]
    frame = convolve(frame, kernel, mode='constant', cval=0.0)
    psf_grid.data[j] = frame

end = time.perf_counter()
print(f"Elapsed time: {end - start:0.2f} seconds")

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
pos = photometry.optimize_position(d_im[x[0]:x[-1]+1, y[0]:y[-1]+1], d_im_inv_var[x[0]:x[-1]+1, y[0]:y[-1]+1], psf_grid, xx, yy, (x0 + dx_sub, y0 + dy_sub))

print("pos - (dx_int, dy_int)", pos[0]-dx_int, pos[1]-dy_int)

z = psf_grid.evaluate(yy, xx, 1.0, pos[1], pos[0])
z /= np.sum(z)

flux, err_flux = photometry.fit_psf(z, d_im[x[0]:x[-1]+1, y[0]:y[-1]+1], d_im_inv_var[x[0]:x[-1]+1, y[0]:y[-1]+1])

print(f"Flux = {flux} +/- {err_flux}.")

fig, ax = plt.subplots(1, 3, figsize=(20, 6))

d_min = np.min(d_im[x[0]:x[-1]+1,y[0]:y[-1]+1])
d_max = np.max(d_im[x[0]:x[-1]+1,y[0]:y[-1]+1])

cm1 = ax[0].imshow(z, origin='lower', extent=[y[0]-0.5, y[-1]+0.5, x[0]-0.5, x[-1]+0.5])
cm2 = ax[1].imshow(d_im[x[0]:x[-1]+1, y[0]:y[-1]+1], origin='lower', vmin=-50, vmax=50, extent=[y[0]-0.5, y[-1]+0.5, x[0]-0.5, x[-1]+0.5])
_ = ax[2].imshow(d_im[x[0]:x[-1]+1, y[0]:y[-1]+1] - flux*z, origin='lower', vmin=-50, vmax=50, extent=[y[0]-0.5, y[-1]+0.5, x[0]-0.5, x[-1]+0.5])

ax[0].title.set_text('PSF')
ax[1].title.set_text('Difference Image')
ax[2].title.set_text('Residual')

plt.colorbar(cm1, ax=ax, location="left")
plt.colorbar(cm2, ax=ax, location="right")

plt.savefig(f"{config_data['output_dir']}/phot_test_single.png")


end = time.perf_counter()
print(f"Elapsed time: {end - start:0.2f} seconds")



#
# Test of fitting PSFs to image stack
#

print("Multiple image test ...")

files = [f"{config_data['output_dir']}/{f}" for f in os.listdir(config_data["output_dir"]) if
         f"d_02_{config_data['data_root']}" in f and f.endswith(".fits")]
files.sort()

#images = [photometry.Image(f) for f in files if "t0143" in f or "t0144" in f or "t0145" in f]
images = [photometry.Image(f) for f in files if "t014" in f]

# These must be int not float
initial_ref_pos = (459, 660)
rad = 5

ref_pos = photometry.optimize_position_stack(images, initial_ref_pos, psf_grid, rad)
#pos = (458.83, 660.09)


fig, ax = plt.subplots(len(images), 3, figsize=(20, len(images)*6 + 1))

x0 = ref_pos[0]
y0 = ref_pos[1]

flux = np.zeros(len(images))
err_flux = np.zeros(len(images))

for i, im in enumerate(images):

    xpos = x0 + im.dx_int + im.dx_sub
    ypos = y0 + im.dy_int + im.dy_sub

    xgrid = np.rint(xpos).astype(int)
    ygrid = np.rint(ypos).astype(int)

    x = np.arange(xgrid - rad, xgrid + rad + 1)
    y = np.arange(ygrid - rad, ygrid + rad + 1)
    xx, yy = np.meshgrid(x, y)

    pos = (xpos, ypos)

    print(pos, xgrid, ygrid)

    z = psf_grid.evaluate(yy, xx, 1.0, pos[1], pos[0])
    z /= np.sum(z)

    flux[i], err_flux[i] = photometry.fit_psf(z, im.data[x[0]:x[-1]+1, y[0]:y[-1]+1], im.inv_var[x[0]:x[-1]+1, y[0]:y[-1]+1])

    d_max = 0.5*np.max(im.data[x[0]:x[-1] + 1, y[0]:y[-1] + 1])
    cm1 = ax[i, 0].imshow(z, origin='lower', extent=[y[0]-0.5, y[-1]+0.5, x[0]-0.5, x[-1]+0.5])
    cm2 = ax[i, 1].imshow(im.data[x[0]:x[-1]+1, y[0]:y[-1]+1], origin='lower', vmin=-d_max, vmax=d_max, extent=[y[0]-0.5, y[-1]+0.5, x[0]-0.5, x[-1]+0.5])
    _ = ax[i, 2].imshow(im.data[x[0]:x[-1]+1, y[0]:y[-1]+1] - flux[i]*z, origin='lower', vmin=-d_max, vmax=d_max, extent=[y[0]-0.5, y[-1]+0.5, x[0]-0.5, x[-1]+0.5])

    plt.colorbar(cm1, ax=ax[i], location="left")
    plt.colorbar(cm2, ax=ax[i], location="right")

    print(im.f_name, im.dx_int, im.dy_int)

ax[0, 0].title.set_text('PSF')
ax[0, 1].title.set_text('Difference Image')
ax[0, 2].title.set_text('Residual')

plt.savefig(f"{config_data['output_dir']}/phot_test_multiple.png")

t = np.arange(len(images))*15

fig, ax = plt.subplots(1, 1, figsize=(12, 6))

ax.errorbar(t, flux, err_flux, fmt=".")

tt = np.linspace(t[0], t[-1], 1001)
tE = 200
t0 = 60
u0 = 0.01
tau = (tt - t0) / tE
u = np.sqrt(u0**2 + tau**2)
A = (u**2 + 2)/(u*np.sqrt(u0**2 + 4))
A *= 9.5
ax.plot(tt, A, 'r-')

ax.set_xlabel('Time (min)')
ax.set_ylabel('Flux')

plt.savefig(f"{config_data['output_dir']}/phot_test_lightcurve.png")

end = time.perf_counter()
print(f"Elapsed time: {end - start:0.2f} seconds")

