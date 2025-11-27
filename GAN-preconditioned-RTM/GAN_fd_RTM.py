"""
frequency-domain reverse time migration
use OFD methods for forward modeling
GAN-based preconditioners for RTM
illumination-compensated cross-correlation imaging conditions
the whole PML absorbing condition
final revision: 2024.8.28
"""
#Taking the Sunken model as an example
import torch
import math
from ofd_matrix_4 import matrix_ofd4, gener_ofdsrc
import numpy as np
from fun_t_filter import fun_t_filter
from bicgstab import bicgstab_model
from generator import UNet
##################################################
################## Basic parameters ##############
##################################################
# area size
nx = 210
nz = 210
# total numbers of grid points
N = (nx - 2) * (nz - 2)
# spatial step (km)
h = 1/50
# layer number of PML
delta = 10
# record period1
T = 10
# set frequency set (Hz)
freq = np.arange(1, 30, 1)

###### Source parameters ######
# Ricker wavelet dominant frequency
f0 = 10
# let sources locate below the 1 grid of PML religion
js = delta + 1
# % choose the part of row
is_ = np.arange(delta, nx - delta - 2, 5)
# by row rule
s_loc = (nx - 2) * js + is_

###### Receiver parameters ######
# let sources locate at the interface of PML religion
jr = delta
# choose the part of row
ir = np.arange(delta, nx - delta - 2, 1)
# by row rule
r_loc = (nx - 2) * jr + ir

# ##### set velocity model ######
# velocity model 1: Sunken model (ao xian) #
c_mat = np.vstack([2 * np.ones(((nz - 3) // 2 + 1, nx - 2)), 2.5* np.ones(((nz - 3) // 2, nx - 2))])
v_row_loc = math.ceil((nx - 2) / 2)
v_col_loc = math.ceil((nz - 2) / 3)
v_row_delta = math.ceil((nx - 2) / 10)
v_col_delta = math.ceil((nz - 2) / 3)
c_mat[v_row_loc:v_row_loc + v_row_delta, v_col_loc:v_col_loc + v_col_delta] = 2

# flatten by row
c_vec = c_mat.reshape(-1, 1)

# normalized factor
I = np.zeros(N)
I1 = np.zeros(N)
I2 = np.zeros(N)
model = UNet()
model.load_state_dict(torch.load('save_GAN/sunken_netG_weights.pth'))
# Move model to the device specified above
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
########### Forward modeling and imaging ###########
for fk in range(len(freq)):
    # increase the effect of parallel
    ff = freq[fk]

    # # OFD scheme
    A = matrix_ofd4(nx, nz, h, ff, delta, c_vec)
    for sk in range(len(s_loc)):
        # show the process
        print(f'{fk + 1}-th freq, {sk}-th source')
        # set source right hand side term
        # OFD scheme
        s = gener_ofdsrc(ff, f0, N, h, c_vec, s_loc[sk])

        ######## frequency-domain imaging condition #########
        ###################### OFD version ###################
        us = bicgstab_model(A, s, model)
        # record the wavefield record and virtual source
        us_r = us.copy()
        us_r[r_loc] = 0
        vir_s = us - us_r
        # compute the receiver wavefield (for cond. 2 and 3)
        # update the imaging results
        ur = bicgstab_model(A, vir_s, model)

        ###########成像条件########
        I += np.real(ur * np.conj(us) * np.exp(1j * 2 * np.pi * ff * T)) / (2 * np.pi)
        I1 += np.real(ur * np.conj(us) * np.exp(1j * 2 * np.pi * ff * T))
        I2 += np.real(us * np.conj(us))+1e-05
# Reshape results to matrix format
I_mat = I.reshape((nz - 2, nx - 2))
I1_mat = I1.reshape((nz - 2, nx - 2))
I2_mat = I2.reshape((nz - 2, nx - 2))
#带小正数归一化
nI_mat = I1_mat / I2_mat

# Laplacian filter
ftI_mat = fun_t_filter(I_mat)
ftnI_mat = fun_t_filter(nI_mat)

# Save results
np.save('2sbI_mat_1.npy', I_mat)
np.save('2sbnI_mat_1.npy', nI_mat)

#
# fig, axs = plt.subplots(1, 2, figsize=(8, 4))
# im2 = axs[0].imshow(I_mat[15:nz - 15, 15:nx - 15], cmap='gray')
# axs[0].set_title('Ic_mat original Image')
# im1 = axs[1].imshow(ftI_mat[15:nz - 15, 15:nx - 15], cmap='gray')
# axs[1].set_title('filtered Image')
# plt.show()
#
# fig, axs = plt.subplots(1, 2, figsize=(8, 4))
# im2 = axs[0].imshow(nI_mat[15:nz - 15, 15:nx - 15], cmap='gray')
# axs[0].set_title('nIc_mat original Image')
# im1 = axs[1].imshow(ftnI_mat[15:nz - 15, 15:nx - 15], cmap='gray')
# axs[1].set_title('filtered Image')
# plt.show()
