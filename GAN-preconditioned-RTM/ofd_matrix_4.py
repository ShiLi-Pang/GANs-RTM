import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import warnings

warnings.filterwarnings('ignore')


def matrix_ofd4(nx, nz, h, f, delta, c_vec):
    # formulate the impedance matrix of velocity model by 4-order OFD method
    # 2D rectangle area
    # implement PML condition all around: 4 directions
    # last revision: 2024.4.20
    # parameter list
    # omega: angular frequency
    # h: spatial step
    # nx: number of nodes along x direction
    # nz: number of nodes along x direction
    # delta: PML layers
    # imaginary unit
    ii = 1j

    # problem scale
    N = (nx - 2) * (nz - 2)

    # nonzero elements of impedance matrix
    spn = 9 * (nx - 6) * (nz - 6) + 16 * (nx + nz - 12) + 14 * (nx + nz - 12) + 96

    # sparse store: vector space
    ai = np.zeros(spn, dtype=int)
    aj = np.zeros(spn, dtype=int)
    as_ = np.zeros(spn, dtype=complex)
    pa = 0

    # set angular frequency
    omega = 2 * np.pi * f

    # PML parameter: the ratio of reflection
    R = 1e-3

    for k in range(N):
        # grid coordinate conversion: row rule
        # 网格坐标转换：行规则
        j = k // (nx - 2)
        i = k - (nx - 2) * j

        # set velocity: row rule
        c = c_vec[k]

        # set PML attenuation function
        if i < delta - 1:
            dx = -3 * c / (2 * delta * h) * np.log(R) * ((delta - i - 1) / delta) ** 2

            dxp = -3 * c / (delta * h) ** 2 * np.log(R) * (-(delta - i - 1) / delta)
        elif i > nx - delta - 2:
            dx = -3 * c / (2 * delta * h) * np.log(R) * ((i - nx + delta + 2) / delta) ** 2

            dxp = -3 * c / (delta * h) ** 2 * np.log(R) * ((i - nx + delta + 2) / delta)
        else:
            dx = 0
            dxp = 0

        if j < delta - 1:
            dz = -3 * c / (2 * delta * h) * np.log(R) * ((delta - j - 1) / delta) ** 2
            dzp = -3 * c / (delta * h) ** 2 * np.log(R) * (-(delta - j - 1) / delta)
        elif j > nz - delta - 2:
            dz = -3 * c / (2 * delta * h) * np.log(R) * ((j - nz + delta + 2) / delta) ** 2
            dzp = -3 * c / (delta * h) ** 2 * np.log(R) * ((j - nz + delta + 2) / delta)
        else:
            dz = 0
            dzp = 0

        tx = 1 - ii * dx / omega
        tz = 1 - ii * dz / omega

        # matlab rule: index start from 1
        # matlab规则：指数从1开始
        kt = k + 1

        # 设置关于波场值 u 的行
        # left1
        if i != 0:
            ai[pa] = kt
            aj[pa] = kt - 1
            as_[pa] = 4 / (3 * tx ** 2) - 2 * ii * dxp * h / (3 * omega * tx ** 3)
            pa += 1

        # left2
        if i > 1:
            ai[pa] = kt
            aj[pa] = kt - 2
            as_[pa] = -1 / (12 * tx ** 2) + ii * dxp * h / (12 * omega * tx ** 3)
            pa += 1

        # right1
        if i != nx - 3:
            ai[pa] = kt
            aj[pa] = kt + 1
            as_[pa] = 4 / (3 * tx ** 2) + 2 * ii * dxp * h / (3 * omega * tx ** 3)
            pa += 1

        # right2
        if i < nx - 4:
            ai[pa] = kt
            aj[pa] = kt + 2
            as_[pa] = -1 / (12 * tx ** 2) - ii * dxp * h / (12 * omega * tx ** 3)
            pa += 1

        # up1
        if j != 0:
            ai[pa] = kt
            aj[pa] = kt - nx + 2
            as_[pa] = 4 / (3 * tz ** 2) - 2 * ii * dzp * h / (3 * omega * tz ** 3)
            pa += 1

        # up2
        if j > 1:
            ai[pa] = kt
            aj[pa] = kt - 2 * nx + 4
            as_[pa] = -1 / (12 * tz ** 2) + ii * dzp * h / (12 * omega * tz ** 3)
            pa += 1

        # down1
        if j != nz - 3:
            ai[pa] = kt
            aj[pa] = kt + nx - 2
            as_[pa] = 4 / (3 * tz ** 2) + 2 * ii * dzp * h / (3 * omega * tz ** 3)
            pa += 1

        # down2
        if j < nz - 4:
            ai[pa] = kt
            aj[pa] = kt + 2 * nx - 4
            as_[pa] = -1 / (12 * tz ** 2) - ii * dzp * h / (12 * omega * tz ** 3)
            pa += 1

        # inner
        ai[pa] = kt
        aj[pa] = kt
        as_[pa] = (omega * h / c) ** 2 - 5 / 2 * (1 / tx ** 2 + 1 / tz ** 2)
        pa += 1

    A = csr_matrix((as_, (ai - 1, aj - 1)), shape=(N, N))

    return A


def gener_ofdsrc(f, f0, N, h, c_vec, s_loc):
    """
    Generate the right hand vector.
    Last revision: 2021.4.20
    """
    # Imaginary unit
    ii = 1j

    # source location (count from 0)
    s0 = s_loc

    # source velocity
    sc = c_vec[s0][0]

    # Amplitude and phase
    t0 = 0.00
    Amp = 1e+5

    s = np.sqrt(2) * Amp / (np.pi * f0) * (f / f0) ** 2 * np.exp(-(f / f0) ** 2) * \
        sp.csr_matrix(([-h ** 2 / sc ** 2 * np.exp(-ii * 2 * np.pi * f * t0)], (np.array([s0]), np.array([0]))),shape=(N, 1))

    return s
