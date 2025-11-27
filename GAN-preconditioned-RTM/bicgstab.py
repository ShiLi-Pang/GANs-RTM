import torch
import sys
from numpy.core._multiarray_umath import inner
sys.path.append('../input')
sys.path.insert(0, 'Utilities')
import numpy as np
from numpy.linalg import norm
try:
    from scipy.sparse.linalg.isolve.utils import make_system
except:
    from scipy.sparse.linalg._isolve.utils import make_system
import warnings

warnings.filterwarnings('ignore')
nx = 210
nz = 210
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def predict(model, p):
    model.eval()
    tensor = np.zeros((1, 2, nz-2, nx-2))
    tensor[0,0] = p.real.reshape(nz-2, nx-2)
    tensor[0,1] = p.imag.reshape(nz-2, nx-2)
    power = np.max(np.abs(tensor))
    # Convert numpy array to torch tensor
    tensor = tensor/power
    tensor = torch.from_numpy(tensor).float().to(device)
    # Forward pass and get the output probabilities
    with torch.no_grad():
        output = model(tensor).cpu().numpy()
    pp = output[0,0] + 1j*output[0,1]
    matrix_p = (power*pp).reshape((nz-2)*(nx-2),)
    return matrix_p

def predictfig(model, p):
    model.eval()
    tensor = np.zeros((1, 2, nz-2, nx-2))
    tensor[0,0] = p.real.reshape(nz-2, nx-2)
    tensor[0,1] = p.imag.reshape(nz-2, nx-2)
    power = np.max(np.abs(tensor))
    # Convert numpy array to torch tensor
    tensor = tensor/power
    tensor = torch.from_numpy(tensor).float().to(device)
    # Forward pass and get the output probabilities
    with torch.no_grad():
        output = model(tensor).cpu().numpy()
    pp = output[0,0] + 1j*output[0,1]
    matrix_p = (power*pp)
    return matrix_p

def bicgstab_model(A, b, mymodel, tol=None, x0=None, it_max=None, M=None):
    (A, _, x, b, postprocess) = make_system(A, None, x0, b)

    if tol == None:
        tol = 1e-5
    if it_max == None:
        it_max = 500
    matvec = A.matvec
    r = b - matvec(x)
    residuals = []
    normb = norm(b)
    normr = norm(r) / normb
    residuals.append(normr)
    get_res = lambda x: norm(matvec(x) - b) / norm(b)
    if normb == 0.0:
        normb = 1.0
    if normr < tol * normb:
        return (x, get_res(x), 0, residuals)
    r_copy = r.copy()
    p = r.copy()
    delta = (inner(r_copy.conjugate(), r))
    it = 0
    while True:

        matrix_p = predict(mymodel, p)
        A_matrix_p = matvec(matrix_p)
        alpha = (delta / inner(r_copy.conjugate(), A_matrix_p))
        s = r - alpha * A_matrix_p

        matrix_s = predict(mymodel, s)
        A_matrix_s = matvec(matrix_s)
        omega = inner(A_matrix_s.conjugate(), s) / inner(A_matrix_s.conjugate(), A_matrix_s)
        x = (x + alpha * matrix_p + omega * matrix_s)
        r = (s - omega * A_matrix_s)
        delta_new = (inner(r_copy.conjugate(), r))
        if (delta == 0):
            print('Error: delta==0!')
            return (x, get_res(x), it, residuals)
        if (omega == 0):
            print('Error: omega==0!')
            return (x, get_res(x), it, residuals)
        beta = ((delta_new / delta) * (alpha / omega))
        delta = (delta_new)
        p = r + beta * (p - omega * A_matrix_p)
        it += 1
        normr = (norm(r) / normb)
        residuals.append(normr)
        if normr < tol:
            return (x, get_res(x), it, residuals)
        if it == it_max:
            return (x, get_res(x), it_max, residuals)
        pass
    pass