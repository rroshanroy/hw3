import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous, _singularize, refineF

# Insert your package here


"""
Q2.2: Seven Point Algorithm for calculating the fundamental matrix
    Input:  pts1, 7x2 Matrix containing the corresponding points from image1
            pts2, 7x2 Matrix containing the corresponding points from image2
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated 3x3 fundamental matrixes.
    
    HINTS:
    (1) Normalize the input pts1 and pts2 scale paramter M.
    (2) Setup the seven point algorithm's equation.
    (3) Solve for the least square solution using SVD. 
    (4) Pick the last two colum vector of vT.T (the two null space solution f1 and f2)
    (5) Use the singularity constraint to solve for the cubic polynomial equation of  F = a*f1 + (1-a)*f2 that leads to 
        det(F) = 0. Solving this polynomial will give you one or three real solutions of the fundamental matrix. 
        Use np.polynomial.polynomial.polyroots to solve for the roots
    (6) Unscale the fundamental matrixes and return as Farray
"""


def _linear_comb(alpha, F1, F2):
    return np.linalg.det(alpha * F1 + (1-alpha) * F2)

def sevenpoint(pts1, pts2, M):
    Farray = []
    # ----- TODO -----
    # YOUR CODE HERE

    num_pts = pts1.shape[0]
    T = np.zeros((3,3))
    T[0,0] = 1./M
    T[1,1] = 1./M
    T[2,2] = 1.

    pts1 = pts1 / M
    pts2 = pts2 / M
    x1 = pts1[:,0]
    x2 = pts2[:,0]
    y1 = pts1[:,1]
    y2 = pts2[:,1]

    A = np.ones((num_pts, 9))
    A[:, 0] = x2 * x1
    A[:, 1] = x2 * y1
    A[:, 2] = x2
    A[:, 3] = y2 * x1
    A[:, 4] = y2 * y1
    A[:, 5] = y2
    A[:, 6] = x1
    A[:, 7] = y1

    _, _, V_T = np.linalg.svd(A)
    F1 = V_T[-2, :].reshape((3,3))
    F2 = V_T[-1, :].reshape((3,3))

    # calculate co-efficients
    t1 = _linear_comb(0, F1, F2)
    t2 = _linear_comb(1, F1, F2)
    t3 = _linear_comb(-1, F1, F2)
    t4 = _linear_comb(2, F1, F2)
    t5 = _linear_comb(-2, F1, F2)
    t6 = _linear_comb(3, F1, F2)
    d = t1
    b = (t2 + t3 - 2*d)/2
    a = (2*t6 - 3*t4 - 6*b + d)/30
    c = (t4 - d - 4*b - 8*a)/2

    # d1 = t1
    # c1 = (2/3)*(t2 - t3) - ((t4-t5)/12)
    # b1 = 0.5*t2 + 0.5*t3 - t1
    # a1 = (-1/6)*(t2 - t3) + (t4-t5)/12

    roots = np.roots([a,b,c,d])
    real_roots = roots[np.isreal(roots)]

    for root in real_roots:
        F = root*F1 + (1-root)*F2
        F = _singularize(F)
        F = refineF(F, pts1, pts2)
        F_norm = np.matmul(T.T, np.matmul(F, T))
        F_norm = F_norm/F_norm[2,2]
        Farray.append(F_norm)

    return Farray


if __name__ == "__main__":
    correspondence = np.load("data/some_corresp.npz")  # Loading correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    pts1, pts2 = correspondence["pts1"], correspondence["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")

    # indices = np.arange(pts1.shape[0])
    # indices = np.random.choice(indices, 7, False)
    indices = np.array([82, 19, 56, 84, 54, 24, 18])

    M = np.max([*im1.shape, *im2.shape])

    Farray = sevenpoint(pts1[indices, :], pts2[indices, :], M)

    print(Farray)

    F = Farray[0]

    np.savez("results/q2_2", F, M)

    # fundamental matrix must have rank 2!
    # assert(np.linalg.matrix_rank(F) == 2)
    # displayEpipolarF(im1, im2, F)

    # Simple Tests to verify your implementation:
    # Test out the seven-point algorithm by randomly sampling 7 points and finding the best solution.
    np.random.seed(1)  # Added for testing, can be commented out

    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)

    max_iter = 500
    pts1_homo = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2_homo = np.hstack((pts2, np.ones((pts2.shape[0], 1))))

    ress = []
    F_res = []
    choices = []
    M = np.max([*im1.shape, *im2.shape])
    for i in range(max_iter):
        choice = np.random.choice(range(pts1.shape[0]), 7)
        pts1_choice = pts1[choice, :]
        pts2_choice = pts2[choice, :]
        Fs = sevenpoint(pts1_choice, pts2_choice, M)
        for F in Fs:
            choices.append(choice)
            res = calc_epi_error(pts1_homo, pts2_homo, F)
            F_res.append(F)
            ress.append(np.mean(res))

    min_idx = np.argmin(np.abs(np.array(ress)))
    F = F_res[min_idx]
    print("Error:", ress[min_idx])

    assert F.shape == (3, 3)
    assert F[2, 2] == 1
    assert np.linalg.matrix_rank(F) == 2
    assert np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1

    displayEpipolarF(im1, im2, F)
