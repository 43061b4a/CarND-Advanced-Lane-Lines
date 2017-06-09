import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


def get_perspective_transform(image, src_in=None, dst_in=None, display=True):
    img_size = image.shape
    if src_in is None:
        src = np.array([[585. / 1280. * img_size[1], 455. / 720. * img_size[0]],
                        [705. / 1280. * img_size[1], 455. / 720. * img_size[0]],
                        [1130. / 1280. * img_size[1], 720. / 720. * img_size[0]],
                        [190. / 1280. * img_size[1], 720. / 720. * img_size[0]]], np.float32)
    else:
        src = src_in

    if dst_in is None:
        dst = np.array([[300. / 1280. * img_size[1], 100. / 720. * img_size[0]],
                        [1000. / 1280. * img_size[1], 100. / 720. * img_size[0]],
                        [1000. / 1280. * img_size[1], 720. / 720. * img_size[0]],
                        [300. / 1280. * img_size[1], 720. / 720. * img_size[0]]], np.float32)
    else:
        dst = dst_in

    warp_m = cv2.getPerspectiveTransform(src, dst)
    warp_minv = cv2.getPerspectiveTransform(dst, src)

    if display:
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        colors = ['r+', 'g+', 'b+', 'c+']
        for i in range(4):
            plt.plot(src[i, 0], src[i, 1], colors[i])

        im2 = cv2.warpPerspective(image, warp_m, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
        plt.subplot(1, 2, 2)
        plt.imshow(im2, cmap='gray')
        for i in range(4):
            plt.plot(dst[i, 0], dst[i, 1], colors[i])

        file_name = "./output_images/perspective_transform.png"
        plt.savefig(file_name)
        plt.show()
        plt.close('all')

    return warp_m, warp_minv


if __name__ == "__main__":
    img = mpimg.imread('./test_images/straight_lines2.jpg')
    get_perspective_transform(img, display=True)
