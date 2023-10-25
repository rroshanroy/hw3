import numpy as np
import matplotlib.pyplot as plt

from helper import _epipoles

from q2_1_eightpoint import eightpoint
from scipy.ndimage import gaussian_filter

# Insert your package here


# Helper functions for this assignment. DO NOT MODIFY!!!
def epipolarMatchGUI(I1, I2, F):
    e1, e2 = _epipoles(F)

    sy, sx, _ = I2.shape

    f, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 9))
    ax1.imshow(I1)
    ax1.set_title("Select a point in this image")
    ax1.set_axis_off()
    ax2.imshow(I2)
    ax2.set_title(
        "Verify that the corresponding point \n is on the epipolar line in this image"
    )
    ax2.set_axis_off()

    while True:
        plt.sca(ax1)
        # x, y = plt.ginput(1, mouse_stop=2)[0]

        out = plt.ginput(1, timeout=3600, mouse_stop=2)

        if len(out) == 0:
            print(f"Closing GUI")
            break

        x, y = out[0]

        xc = int(x)
        yc = int(y)
        v = np.array([xc, yc, 1])
        l = F.dot(v)
        s = np.sqrt(l[0] ** 2 + l[1] ** 2)

        if s == 0:
            print("Zero line vector in displayEpipolar")

        l = l / s

        if l[0] != 0:
            ye = sy - 1
            ys = 0
            xe = -(l[1] * ye + l[2]) / l[0]
            xs = -(l[1] * ys + l[2]) / l[0]
        else:
            xe = sx - 1
            xs = 0
            ye = -(l[0] * xe + l[2]) / l[1]
            ys = -(l[0] * xs + l[2]) / l[1]

        # plt.plot(x,y, '*', 'MarkerSize', 6, 'LineWidth', 2);
        ax1.plot(x, y, "*", markersize=6, linewidth=2)
        ax2.plot([xs, xe], [ys, ye], linewidth=2)

        # draw points
        x2, y2 = epipolarCorrespondence(I1, I2, F, xc, yc)
        ax2.plot(x2, y2, "ro", markersize=8, linewidth=2)
        plt.draw()


"""
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2
            
    Hints:
    (1) Given input [x1, x2], use the fundamental matrix to recover the corresponding epipolar line on image2
    (2) Search along this line to check nearby pixel intensity (you can define a search window) to 
        find the best matches
    (3) Use guassian weighting to weight the pixel simlairty

"""


def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation
    # ----- TODO -----
    # YOUR CODE HERE

    # Given input [x1, x2], use the fundamental matrix to recover the corresponding epipolar line on image2
    hom_x1 = np.array([x1, y1, 1], dtype=float)
    a, b, c = np.matmul(F, hom_x1)
    # print(a, b, c)
    #ax + by + c = 0
    
    # (2) Search along this line to check nearby pixel intensity (you can define a search window) to  find the best matches
    h = im1.shape[0]
    w = im1.shape[1]

    # create x1 window of search
    win = 5
    win_x1 = x1 + win
    win_x2 = x1 - win
    win_y1 = y1 + win
    win_y2 = y1 - win
    target_crop = im1[win_y2:win_y1, win_x2:win_x1]
    #window_x1 = im1[y1 - win:y1 + win, x1 - win:x1 + win]

    slope = -a/b
    img_slope = h/w
    space = 40
    if np.absolute(slope) < img_slope:
        x2_start = max(x1-space, win_x2)
        x2_end = min(x1+space, win_x1)

        x_range = np.arange(x2_start, x2_end, 1)
        min_err = np.inf
        min_x, min_y = None, None
        for x in x_range:
            y = int(-(c + a*x)/b)
            # print(f"x: {x}, y: {y}")

            w_x1 = int(x + win)
            w_x2 = int(x - win)
            w_y1 = int(y + win)
            w_y2 = int(y - win)

            cur_crop = im2[w_y2:w_y1, w_x2:w_x1]

            delta_window = cur_crop - target_crop
            error = gaussian_filter(np.absolute(delta_window).mean(), sigma=31)
            # print(f"Error: {error}, x2: {(w_x1, w_y1)}, x1: {(w_x2, w_y2)}")

            if error < min_err:
                min_err = error
                min_x = x
                min_y = y

    else :
        y2_start = y1-space
        y2_end = y1+space

        y_range = np.arange(y2_start, y2_end, 1)
        min_err = np.inf
        min_x, min_y = None, None
        for y in y_range:
            x = int(-(c + b*y)/a)
            # print(f"x: {x}, y: {y}")  

            w_x1 = int(x + win)
            w_x2 = int(x - win)
            w_y1 = int(y + win)
            w_y2 = int(y - win)

            cur_crop = im2[w_y2:w_y1, w_x2:w_x1]

            delta_window = cur_crop - target_crop
            error = gaussian_filter(np.absolute(delta_window).mean(), sigma=31)
            # print(f"Error: {error}, x2: {(w_x1, w_y1)}, x1: {(w_x2, w_y2)}")

            if error < min_err:
                min_err = error
                min_x = x
                min_y = y

    return min_x, min_y


if __name__ == "__main__":
    correspondence = np.load("data/some_corresp.npz")  # Loading correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    pts1, pts2 = correspondence["pts1"], correspondence["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")

    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))

    np.savez("results/q4_1.npz", F, pts1, pts2)
    epipolarMatchGUI(im1, im2, F)

    # Simple Tests to verify your implementation:
    x2, y2 = epipolarCorrespondence(im1, im2, F, 119, 217)
    assert np.linalg.norm(np.array([x2, y2]) - np.array([118, 181])) < 10
