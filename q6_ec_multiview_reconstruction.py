import numpy as np
import matplotlib.pyplot as plt

import os

from helper import visualize_keypoints, plot_3d_keypoint, connections_3d, colors
from q3_2_triangulate import triangulate

# Insert your package here

"""
Q6.1 Multi-View Reconstruction of keypoints.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx3 matrix with the 2D image coordinates and confidence per row
            C2, the 3x4 camera matrix
            pts2, the Nx3 matrix with the 2D image coordinates and confidence per row
            C3, the 3x4 camera matrix
            pts3, the Nx3 matrix with the 2D image coordinates and confidence per row
    Output: P, the Nx3 matrix with the corresponding 3D points for each keypoint per row
            err, the reprojection error.

Modified by Vineet Tambe, 2023.
"""


def MultiviewReconstruction(C1, pts1, C2, pts2, C3, pts3, Thres=100):
    # TODO: Replace pass by your implementation
    conf_mask1 = (pts1[..., -1] > Thres).astype(int)
    conf_mask2 = (pts2[..., -1] > Thres).astype(int)
    conf_mask3 = (pts3[..., -1] > Thres).astype(int)

    conf_mask = (conf_mask1 + conf_mask2 + conf_mask3) >= 2
    p1_1, p1_2, p1_3 = C1[0], C1[1], C1[2]
    p2_1, p2_2, p2_3 = C2[0], C2[1], C2[2]
    p3_1, p3_2, p3_3 = C3[0], C3[1], C3[2]

    conf_pts1 = pts1[conf_mask]
    conf_pts2 = pts2[conf_mask]
    conf_pts3 = pts3[conf_mask]

    error = 0
    i=0
    P = np.zeros((len(conf_pts1), 3))

    for pt1, pt2, pt3 in zip(conf_pts1, conf_pts2, conf_pts3):
        A = np.empty((0,4))
        if pt1[-1]:
             A = np.vstack((A, np.array([pt1[1] * p1_3 - p1_2, p1_1 - pt1[0] * p1_3])))
             #np.concatenate([A, np.array([pt1[1] * p1_3 - p1_2, p1_1 - pt1[0] * p1_3])[np.newaxis, :]])#[np.newaxis, :]
        if pt2[-1]:
            A = np.vstack((A, np.array([pt2[1] * p2_3 - p2_2, p2_1 - pt2[0] * p2_3])))
            # A = np.concatenate([A, np.array([pt2[1] * p2_3 - p2_2, p2_1 - pt2[0] * p2_3])[np.newaxis, :]])#[np.newaxis, :]
        if pt3[-1]:
            A = np.vstack((A, np.array([pt3[1] * p3_3 - p3_2, p3_1 - pt3[0] * p3_3])))
            #A = np.concatenate([A, np.array([pt3[1] * p3_3 - p3_2, p3_1 - pt3[0] * p3_3])[np.newaxis, :]])#[np.newaxis, :]

        _, _, V_T = np.linalg.svd(A, 0)
        X = V_T[-1]  # extract last row from V_T matrix or last column from V matrix
        X = X/X[-1]

        # calculate reprojection error
        projx1 = np.matmul(C1, X)  # or with X.T??
        projx2 = np.matmul(C2, X)
        projx3 = np.matmul(C3, X)
        projx1 = (projx1 / projx1[-1])[:-1]
        projx2 = (projx2 / projx2[-1])[:-1]
        projx3 = (projx3 / projx3[-1])[:-1]

        err = 0
        if pt1[-1]:
            err += np.linalg.norm(projx1 - pt1[:-1])**2
        if pt2[-1]:
            err+= np.linalg.norm(projx2 - pt2[:-1])**2  
        if pt3[-1]:
            err+= np.linalg.norm(projx3 - pt3[:-1])**2

        P[i] = X[:-1]
        i +=1

        err = err/len(A)  # divide by 2????
        error += err

    return P, error

"""
Q6.2 Plot Spatio-temporal (3D) keypoints
    :param car_points: np.array points * 3
"""


def plot_3d_keypoint_video(pts_3d_video):
    # TODO: Replace pass by your implementation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    f = len(pts_3d_video)
    for i in range(f):
        pts_3d = pts_3d_video[i]
        #num_pts = len(pts_3d)#.shape[0]
        for j in range(len(connections_3d)):
            i0, i1 = connections_3d[j]
            xline = [pts_3d[i0, 0], pts_3d[i1, 0]]
            yline = [pts_3d[i0, 1], pts_3d[i1, 1]]
            zline = [pts_3d[i0, 2], pts_3d[i1, 2]]
            ax.plot(xline, yline, zline, color=colors[j], alpha = i/f)

    np.set_printoptions(threshold=1e6, suppress=True)
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")
    plt.show()


# Extra Credit
if __name__ == "__main__":
    pts_3d_video = []
    for loop in range(10):
        print(f"processing time frame - {loop}")

        data_path = os.path.join("data/q6/", "time" + str(loop) + ".npz")
        image1_path = os.path.join("data/q6/", "cam1_time" + str(loop) + ".jpg")
        image2_path = os.path.join("data/q6/", "cam2_time" + str(loop) + ".jpg")
        image3_path = os.path.join("data/q6/", "cam3_time" + str(loop) + ".jpg")

        im1 = plt.imread(image1_path)
        im2 = plt.imread(image2_path)
        im3 = plt.imread(image3_path)

        data = np.load(data_path)
        pts1 = data["pts1"]
        pts2 = data["pts2"]
        pts3 = data["pts3"]

        K1 = data["K1"]
        K2 = data["K2"]
        K3 = data["K3"]

        M1 = data["M1"]
        M2 = data["M2"]
        M3 = data["M3"]

        # Note - Press 'Escape' key to exit img preview and loop further
        img = visualize_keypoints(im2, pts2)

        # TODO: YOUR CODE HERE
        # TODO: YOUR CODE HERE
        C1 = np.matmul(K1, M1)
        C2 = np.matmul(K2, M2) 
        C3 = np.matmul(K3, M3)
        P, err = MultiviewReconstruction(C1, pts1, C2, pts2, C3, pts3)
        plot_3d_keypoint(P)
        pts_3d_video.append(P)

    plot_3d_keypoint_video(pts_3d_video)

    np.savez("results/q6_1.npz", np.array(pts_3d_video))
