from binary_thresholder import binary_threshold_pipeline
from perspective_transformer import get_perspective_transform

import pickle
import cv2
import glob
import matplotlib.pyplot as plt


def pipeline(img):
    # Undistort
    camera_parameters = pickle.load(open("./calibration_parameters/camera_matrix.p", "rb"))

    img_undistort = cv2.undistort(img, camera_parameters['mtx'],
                                  camera_parameters['dist'], None, camera_parameters['mtx'])

    # Perspective Transform
    warp_m, warp_minv = get_perspective_transform(img_undistort, display=False)
    perspective_transformed = cv2.warpPerspective(img, warp_m, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    binary_img = binary_threshold_pipeline(perspective_transformed)

    return binary_img, warp_minv


def main():
    # Make a list of example images
    images = glob.glob('./test_images/*.jpg')

    # Set up plot
    fig, axs = plt.subplots(len(images), 2, figsize=(10, 20))
    fig.subplots_adjust(hspace=.2, wspace=.001)
    axs = axs.ravel()

    i = 0
    for idx, image in enumerate(images):
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_bin, warp_minv = pipeline(img)
        axs[i].imshow(img)
        axs[i].axis('off')
        i += 1
        axs[i].imshow(img_bin, cmap='gray')
        axs[i].axis('off')
        i += 1

    file_name = "./output_images/test_images_pipeline.png"
    plt.savefig(file_name, bbox_inches='tight')

if __name__ == "__main__":
    main()
