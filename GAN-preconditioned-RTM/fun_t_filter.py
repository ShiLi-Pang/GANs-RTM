import numpy as np

def fun_t_filter(I_mat):
    # test the Laplace filter
    # default: 2-order

    # m=nz-2;
    # n=nx-2;
    m = I_mat.shape[0]
    n = I_mat.shape[1]

    # initalize the filter
    La_r = np.zeros((m, m))
    La_c = np.zeros((n, n))

    # the number of order for discretion
    no = 2

    coeff = [1, -2, 1]

    for i in range(int(no/2) + 1):
        if i == 0:
            La_r += coeff[int(no/2)] * np.diag(np.ones(m))
            La_c += coeff[int(no/2)] * np.diag(np.ones(n))
        else:
            La_r += coeff[int(no/2) + i] * (np.diag(np.ones(m-i), -i) + np.diag(np.ones(m-i), i))
            La_c += coeff[int(no/2) + i] * (np.diag(np.ones(n-i), -i) + np.diag(np.ones(n-i), i))

    # La1_r = np.diag(-2 * np.ones(m)) + np.diag(np.ones(m-1), -1) + np.diag(np.ones(m-1), 1)
    # La1_c = np.diag(-2 * np.ones(n)) + np.diag(np.ones(n-1), -1) + np.diag(np.ones(n-1), 1)

    ftI_mat = np.dot(La_r, I_mat) + np.dot(I_mat, La_c)

    # norm(IL-ILt,1)

    # ILt=La_r*Icm+Icm*La_c;

    # np.linalg.norm(La_r - La1_r, ord=1)
    # np.linalg.norm(La_c - La1_c, ord=1)

    return ftI_mat