import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous
from q2_1_eightpoint import eightpoint
from q2_2_sevenpoint import sevenpoint
from q3_2_triangulate import findM2

import scipy

# Insert your package here


# Helper functions for this assignment. DO NOT MODIFY!!!
"""
Helper functions.

Written by Chen Kong, 2018.
Modified by Zhengyi (Zen) Luo, 2021
"""


def plot_3D_dual(P_before, P_after):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Blue: before; red: after")
    ax.scatter(P_before[:, 0], P_before[:, 1], P_before[:, 2], c="blue")
    ax.scatter(P_after[:, 0], P_after[:, 1], P_after[:, 2], c="red")
    while True:
        x, y = plt.ginput(1, mouse_stop=2)[0]
        plt.draw()


"""
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
            nIters, Number of iterations of the Ransac
            tol, tolerence for inliers
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers

    Hints:
    (1) You can use the calc_epi_error from q1 with threshold to calcualte inliers. Tune the threshold based on 
        the results/expected number of inliners. You can also define your own metric. 
    (2) Use the seven point alogrithm to estimate the fundamental matrix as done in q1
    (3) Choose the resulting F that has the most number of inliers
    (4) You can increase the nIters to bigger/smaller values
 
"""


def ransacF(pts1, pts2, M, nIters=1000, tol=10):
    # TODO: Replace pass by your implementation

    max_inliers = 0
    F_best = None
    i = 0
    inliers = None
    while i < nIters:
        idx = np.random.randint(len(pts1), size=8)
        eight_pts1 = pts1[idx]
        eight_pts2 = pts2[idx]

        F = eightpoint(eight_pts1, eight_pts2, M)

        pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)
        error = calc_epi_error(pts1_homogenous, pts2_homogenous, F)
        # minErr, maxnumInliers = err[in_idx].mean(), numInliers,

        # inliers_mask = np.where(error < tol, True, False)
        inliers_mask = error < tol
        #errors[errors < tol].mean()

        if np.sum(inliers_mask) > max_inliers:
            F_best = F
            min_error = error[inliers_mask].mean()
            max_inliers = np.sum(inliers_mask)
            inliers = inliers_mask
        i+=1

        print(f"i: {i}, min_error: {min_error}")

    return F_best, inliers


"""
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
"""


def rodrigues(r):
    # TODO: Replace pass by your implementation
    theta = np.linalg.norm(r)

    if theta == 0:
        R = np.identity(3)
    
    else:
        u = r/theta
        u_x = np.array([[0, -1.*u[2], u[1]],
                       [u[2], 0, -u[0]],
                       [-u[1], u[0], 0]])
        u = u.reshape(3, 1)
        R = np.identity(3)*np.cos(theta) + (1-np.cos(theta))*np.matmul(u, u.T) + u_x*np.sin(theta)

    return R

    

"""
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
"""


def invRodrigues(R):
    # TODO: Replace pass by your implementation
    A =  (R - R.T) / 2
    rho = np.array([A[2,1], A[0,2], A[1,0]])
    s = np.linalg.norm(rho)
    c = (np.trace(R)-1)/2

    S = lambda r: -r if (np.linalg.norm(r)==np.pi and ((r[0]==r[1]==0 and r[3]<0) or (r[0]==0 and r[1]<0) or (r[0]<0))) else r

    if s==0 and c==1:
        r = 0
    elif s==0 and c==-1:
        R_t = R + np.identity(3)  # TODO: check if this is a non-zero column!
        col_norms = np.linalg.norm(R_t, ord=2, axis=0)
        max_norm_column = np.argmax(col_norms)
        v = R_t[:, max_norm_column]
        u = v / np.linalg.norm(v)
        r = S(u*np.pi)
    else:
        u = rho / s
        theta = np.arctan2(s,c)
        r = u * theta

    return r



"""
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
"""


def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # TODO: Replace pass by your implementation
    trans = x[-3:] 
    rot = x[-6:-3]
    P = x[:-6]

    rot_mat = rodrigues(rot)
    #M2 = np.hstack((R2, trans.reshape((3, 1))))
    M2 = np.concatenate([rot_mat, trans.reshape((3, 1))], axis=1)

    C1 = np.matmul(K1, M1)
    C2 = np.matmul(K2, M2)

    P = P.reshape((int(len(P)/3), 3))
    projected_x1 = np.matmul(C1, P.T)
    projected_x1 = projected_x1/projected_x1[-1]
    
    projected_x2 = np.matmul(C2, P.T)
    projected_x2 = projected_x2/projected_x2[-1]


    res_1 = (p1.T - projected_x1[:2])
    res_1 = res_1.reshape([-1])

    res_2 = (p2.T - projected_x2[:2])
    res_2 = res_2.reshape((-1))
    #res_out = np.concatenate((res_1, res_2), axis=0)
    res_out = np.vstack((res_1, res_2))
    
    return res_out


"""
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
            o1, the starting objective function value with the initial input
            o2, the ending objective function value after bundle adjustment

    Hints:
    (1) Use the scipy.optimize.minimize function to minimize the objective function, rodriguesResidual. 
        You can try different (method='..') in scipy.optimize.minimize for best results. 
"""


def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    obj_start = obj_end = 0
    # ----- TODO -----
    # YOUR CODE HERE
    
    return M2, P, obj_start, obj_end


if __name__ == "__main__":
    np.random.seed(1)  # Added for testing, can be commented out

    some_corresp_noisy = np.load(
        "data/some_corresp_noisy.npz"
    )  # Loading correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    noisy_pts1, noisy_pts2 = some_corresp_noisy["pts1"], some_corresp_noisy["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")

    #F, inliers = ransacF(noisy_pts1, noisy_pts2, M=np.max([*im1.shape, *im2.shape]))

    # displayEpipolarF(im1, im2, F)

    # Simple Tests to verify your implementation:
    pts1_homogenous, pts2_homogenous = toHomogenous(noisy_pts1), toHomogenous(
        noisy_pts2
    )

    # assert F.shape == (3, 3)
    # assert F[2, 2] == 1
    # assert np.linalg.matrix_rank(F) == 2

    # Simple Tests to verify your implementation:
    from scipy.spatial.transform import Rotation as sRot

    rotVec = sRot.random()
    mat = rodrigues(rotVec.as_rotvec())

    assert np.linalg.norm(rotVec.as_rotvec() - invRodrigues(mat)) < 1e-3
    assert np.linalg.norm(rotVec.as_matrix() - mat) < 1e-3

    # Visualization:
    np.random.seed(1)
    correspondence = np.load(
        "data/some_corresp_noisy.npz"
    )  # Loading noisy correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    pts1, pts2 = correspondence["pts1"], correspondence["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")
    M = np.max([*im1.shape, *im2.shape])

    # TODO: YOUR CODE HERE
    """
    Call the ransacF function to find the fundamental matrix
    Call the findM2 function to find the extrinsics of the second camera
    Call the bundleAdjustment function to optimize the extrinsics and 3D points
    Plot the 3D points before and after bundle adjustment using the plot_3D_dual function
    """
    F, inliers = ransacF(pts1, pts2, M=np.max([*im1.shape, *im2.shape]), nIters=100, tol=6)
    inliers = inliers.reshape(len(inliers),)

    pts1_, pts2_ = pts1[inliers, :], pts2[inliers, :]
    M2_init, C2, P_init = findM2(F, pts1_, pts2_, intrinsics)
    M1 = np.hstack((np.identity(3), np.zeros(3)[:, np.newaxis]))
    M2, P, obj_start, obj_end = bundleAdjustment(
        K1, M1, pts1_, K2, M2_init, pts2_, P_init)
    print("obj_start:", obj_start, "obj_end:", obj_end)
    plot_3D_dual(P_init, P)
