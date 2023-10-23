import numpy as np
import matplotlib.pyplot as plt

from helper import camera2, toHomogenous
from q2_1_eightpoint import eightpoint
from q3_1_essential_matrix import essentialMatrix

# Insert your package here


"""
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.

    Hints:
    (1) For every input point, form A using the corresponding points from pts1 & pts2 and C1 & C2
    (2) Solve for the least square solution using np.linalg.svd
    (3) Calculate the reprojection error using the calculated 3D points and C1 & C2 (do not forget to convert from 
        homogeneous coordinates to non-homogeneous ones)
    (4) Keep track of the 3D points and projection error, and continue to next point 
    (5) You do not need to follow the exact procedure above. 
"""


def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    # ----- TODO -----
    # YOUR CODE HERE
    pts1 = toHomogenous(pts1)
    pts2 = toHomogenous(pts2)

    p1_1, p1_2, p1_3 = C1[0], C1[1], C1[2]
    p2_1, p2_2, p2_3 = C2[0], C2[1], C2[2]
    #X = np.zeros((len(pts1), 3))

    err = 0
    i=0
    P = np.zeros((len(pts1), 3))
    
    for pt1, pt2 in zip(pts1, pts2):
        A = np.zeros((4,4))
        # constructing A matrix
        A[0] = pt2[1]*p2_3 - p2_2
        A[1] = p2_1 - pt2[0]*p2_3
        A[2] = pt1[1]*p1_3 - p1_2
        A[3] = p1_1 - pt1[0]*p1_3

        _, _, V_T = np.linalg.svd(A, 0)
        X = V_T[-1]  # extract last row from V_T matrix or last column from V matrix
        X = X/X[-1]

        
        # calculate reprojection error
        projx1 = np.matmul(C1, X)
        projx2 = np.matmul(C2, X)
        projx1 = (projx1 / projx1[-1])[:-1]
        projx2 = (projx2 / projx2[-1])[:-1]

        err += np.linalg.norm(projx1 - pt1[:-1])**2 + np.linalg.norm(projx2 - pt2[:-1])**2
        # err += np.linalg.norm(projx1 - pt1)**2 + np.linalg.norm(projx2 - pt2)**2

        P[i] = X[:-1]
        i +=1
    
    return P, err


"""
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
"""


def findM2(F, pts1, pts2, intrinsics, filename="q3_3.npz"):
    """
    Q2.2: Function to find camera2's projective matrix given correspondences
        Input:  F, the pre-computed fundamental matrix
                pts1, the Nx2 matrix with the 2D image coordinates per row
                pts2, the Nx2 matrix with the 2D image coordinates per row
                intrinsics, the intrinsics of the cameras, load from the .npz file
                filename, the filename to store results
        Output: [M2, C2, P] the computed M2 (3x4) camera projective matrix, C2 (3x4) K2 * M2, and the 3D points P (Nx3)

    ***
    Hints:
    (1) Loop through the 'M2s' and use triangulate to calculate the 3D points and projection error. Keep track
        of the projection error through best_error and retain the best one.
    (2) Remember to take a look at camera2 to see how to correctly reterive the M2 matrix from 'M2s'.

    """
    # ----- TODO -----
    # YOUR CODE HERE


    K1, K2 = intrinsics["K1"], intrinsics["K2"]

    E = essentialMatrix(F, K1, K2)
    M2s = camera2(E)
    M1 = np.concatenate([np.identity(3), np.zeros((3,1))], axis=1)
    err_min = np.inf
    P = None

    for i in range(M2s.shape[-1]):
        M2 = M2s[:, :, i]
        C1 = np.matmul(K1, M1)
        C2 = np.matmul(K2, M2)
        Pcur, err = triangulate(C1, pts1, C2, pts2)
        if err < err_min: 
            err_min = err
            P = Pcur

    return M2, C2, P


if __name__ == "__main__":
    correspondence = np.load("data/some_corresp.npz")  # Loading correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    pts1, pts2 = correspondence["pts1"], correspondence["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")

    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))

    M2, C2, P = findM2(F, pts1, pts2, intrinsics)

    # Simple Tests to verify your implementation:
    M1 = np.hstack((np.identity(3), np.zeros(3)[:, np.newaxis]))
    C1 = K1.dot(M1)
    C2 = K2.dot(M2)
    P_test, err = triangulate(C1, pts1, C2, pts2)
    assert err < 500
