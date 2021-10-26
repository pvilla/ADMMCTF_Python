############################################################
#                                                          #
#   TEST ROUTINE FOR PHASE RETRIEVAL SOLVED BY ADMM-TV     #
#                                                          #
############################################################
import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
import ADMMCTF
from skimage import measure
import time

print('\nTEST ROUTINE FOR PHASE RETRIEVAL SOLVED BY ADMM-TV\n')

# Load flat field intensity map phantom

phantom_name = 'test_phantom.mat'
mat_file = sio.loadmat(phantom_name)

# Load detector intensity
img = mat_file["img"]

# print(img.shape)
n, m = img.shape

# Load mask
mask = mat_file["mask"]

# Physical Parameters
E = 17.01  # keV
wvl = 12.4/E*1e-10
pxs = 1e-8  # pixelsize
DOF = pxs**2/wvl
D = DOF*100
betaoverdelta = 5e-1

# ADMM-TV setting
niter = 200  # number of iterations
eps = 1e-3
# stopping threshold
tau = 5e-5  # connection strength
eta = 0.02*tau  # regularization strength
#
phys = 0  # flag for the physical constraints
ks = ADMMCTF.kernel_grad().shape[0]-1  # size of the gradient kernel

print(f"\nADMM settings:")
print(f"wvl = {wvl:.3e})")
print(f"niter max = {niter:d}")
print(f"eps= {eps:.3e}")
print(f"eta = {eta:.3e}")
print(f"tau = {tau:.3e}")
print(f"ks={ks: d}\n")

# Padding image
b = np.pad(img, [ks, ks], mode='edge')

# FPSF(Fourier transformed of the PSF)
FPSF = []
# If FPSF or OTF is required, define it here

# # Display input data
plt.figure(dpi=150)
plt.imshow(b, cmap="gray", aspect="equal")
plt.title('Input intensity')

if np.mean(mask) != 1.0:
    plt.figure(dpi=150)
    plt.imshow(mask, cmap='gray')
    plt.title('Support constraint')


# # Iterative reconstruction
print(f"\nStarting reconstruction with ADMM ....")
tic = time.time()
# # #mdic = {"b_python": b, "niter_python": niter, "eps_python": eps, "eta_python": eta, "tau_python": tau, "phys_python": phys, "mask_python": mask, "D_python": D, "lambda_python": wvl,
# # #        "pxs_python": pxs, "betaoverdelta_python": betaoverdelta, "FPSF_python": FPSF}
# # #sio.savemat("matlab_test.mat", mdic)
# # #sys.exit(0)
x_it = ADMMCTF.admm_ctf_betaoverdelta(
    b, niter, eps, eta, tau, phys, mask, wvl, D, pxs, betaoverdelta, FPSF)
TimeInterval = time.time()-tic
print(f".... reconstruction done!\n")

plt.figure(dpi=150)
plt.imshow(x_it, cmap='gray')
plt.title('Iterative reconstruction')

# # Analytical reconstruction

epsilon = 1e-1
x_an = ADMMCTF.ctf_fixedratio_retrieval(
    b, epsilon, D, wvl, pxs, betaoverdelta, FPSF)
# x_an = x_an(ks+1: n+ks, ks+1: m+ks)
plt.figure(dpi=150)
plt.imshow(x_an, cmap='gray')
plt.title('Analytical reconstruction')

# # Compute mean squared error over the oracle
phase_map = np.angle(mat_file["Uin"])
plt.figure(dpi=150)
plt.imshow(phase_map, cmap='gray')
plt.title('Phantom Phase map')

err_it = measure.compare_psnr(phase_map, x_it)
err_an = measure.compare_psnr(phase_map, x_an[2:-2, 2:-2])
print(f"Iterative reconstruction error (PSNR) : {err_it:.5e}")
print(f"Analytical reconstruction error (PSNR) : {err_an:.5e}")
print(f"Time elapsed: {TimeInterval:.4g} sec.\n\n")
