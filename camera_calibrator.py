# Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.

# Based on code from OpenCV tutorials.

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
import matplotlib.image as mpimg

internal_row_intersect = 6
internal_col_intersect = 9

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def calibrator(original_images):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((internal_row_intersect * internal_col_intersect, 3), np.float32)
    objp[:, :2] = np.mgrid[0:internal_row_intersect, 0:internal_col_intersect].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    for idx, file_name in enumerate(original_images):
        img = cv2.imread(file_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (internal_row_intersect, internal_col_intersect), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (internal_row_intersect, internal_col_intersect), corners2, ret)
            chess_file_name = "./output_images/" + str(idx) + "_chessboard.png"
            cv2.imwrite(chess_file_name, img)

    img = cv2.imread(original_images[0])
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    return mtx, dist


def calibrator_main():
    images = glob.glob('./camera_cal/cali*.jpg')
    test_images = glob.glob('./test_images/*.jpg')
    images.extend(test_images)

    mtx, dist = calibrator(images)
    # undistort
    for idx, fname in enumerate(images):
        img = mpimg.imread(fname)  # to keep things consistent with plotting color space.
        dst = cv2.undistort(img, mtx, dist, None, mtx)

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=50)
        ax2.imshow(dst)
        ax2.set_title('Undistorted Image', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        file_name = "./output_images/" + str(idx) + "_compare.png"
        plt.savefig(file_name)
        plt.close('all')

    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {"mtx": mtx, "dist": dist}
    pickle.dump(dist_pickle, open("calibration_parameters/camera_matrix.p", "wb"))


if __name__ == "__main__":
    calibrator_main()
