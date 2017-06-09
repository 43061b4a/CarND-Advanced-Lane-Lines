import numpy as np
import cv2
import matplotlib.pyplot as plt
from pipeline import pipeline
from lane_drawer import draw_lane


def sliding_window_polyfit(img):
    # Take a histogram of the bottom half of the image
    hist = np.sum(img[img.shape[0] // 2:, :], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(hist.shape[0] // 2)

    leftx_base = np.argmax(hist[:midpoint])
    # rightx_base = np.argmax(hist[midpoint:(midpoint + quarter_point)]) + midpoint
    rightx_base = np.argmax(hist[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 10
    # Set height of windows
    window_height = np.int(img.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 80
    # Set minimum number of pixels found to recenter window
    minpix = 40
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    # Rectangle data for visualization
    rectangle_data = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        rectangle_data.append((win_y_low, win_y_high, win_xleft_low, win_xleft_high, win_xright_low, win_xright_high))
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit, right_fit = (None, None)
    # Fit a second order polynomial to each
    if len(leftx) != 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit = np.polyfit(righty, rightx, 2)

    visualization_data = (rectangle_data, hist)

    return left_fit, right_fit, left_lane_inds, right_lane_inds, visualization_data


# this assumes that the fit will not change significantly from one video frame to the next
def polyfit_using_previous_fit(binary_warped, left_fit_prev, right_fit_prev):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 80
    left_lane_inds = (
        (nonzerox > (left_fit_prev[0] * (nonzeroy ** 2) + left_fit_prev[1] * nonzeroy + left_fit_prev[2] - margin)) &
        (nonzerox < (left_fit_prev[0] * (nonzeroy ** 2) + left_fit_prev[1] * nonzeroy + left_fit_prev[2] + margin)))
    right_lane_inds = (
        (nonzerox > (right_fit_prev[0] * (nonzeroy ** 2) + right_fit_prev[1] * nonzeroy + right_fit_prev[2] - margin)) &
        (nonzerox < (right_fit_prev[0] * (nonzeroy ** 2) + right_fit_prev[1] * nonzeroy + right_fit_prev[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit_new, right_fit_new = (None, None)
    if len(leftx) != 0:
        # Fit a second order polynomial to each
        left_fit_new = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit_new = np.polyfit(righty, rightx, 2)
    return left_fit_new, right_fit_new, left_lane_inds, right_lane_inds


# Method to determine radius of curvature and distance from lane center
def curvature_calculator(binary_image, l_fit, r_fit, l_lane_inds, r_lane_inds):
    # Define conversions in x and y from pixels space to meters
    # ym_per_pix = 3.048 / 100

    ym_per_pix = 30 / 720
    xm_per_pix = 3.7 / 700

    left_curverad, right_curverad, center_dist = (0, 0, 0)
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    h = binary_image.shape[0]
    ploty = np.linspace(0, h - 1, h)
    y_eval = np.max(ploty)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Again, extract left and right line pixel positions
    leftx = nonzerox[l_lane_inds]
    lefty = nonzeroy[l_lane_inds]
    rightx = nonzerox[r_lane_inds]
    righty = nonzeroy[r_lane_inds]

    if len(leftx) != 0 and len(rightx) != 0:
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = (
                             (1 + (
                                 2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[
                                     1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])
        # Now our radius of curvature is in meters

    # Distance from center is image x midpoint - mean of l_fit and r_fit intercepts
    if r_fit is not None and l_fit is not None:
        car_position = binary_image.shape[1] / 2
        l_fit_x_int = l_fit[0] * h ** 2 + l_fit[1] * h + l_fit[2]
        r_fit_x_int = r_fit[0] * h ** 2 + r_fit[1] * h + r_fit[2]
        lane_center_position = (r_fit_x_int + l_fit_x_int) / 2
        center_dist = (car_position - lane_center_position) * xm_per_pix

    # Using US regulations data.
    # http://onlinemanuals.txdot.gov/txdotmanuals/rdw/horizontal_alignment.htm#BGBHGEGC
    # 4575ft = 1394.46m
    left_curverad = min(left_curverad, 1395)
    right_curverad = min(right_curverad, 1395)

    return left_curverad, right_curverad, center_dist


if __name__ == "__main__":
    # visualize the result on example image
    example_img = cv2.imread('./test_images/test2.jpg')
    example_img = cv2.cvtColor(example_img, cv2.COLOR_BGR2RGB)
    example_img_bin, minv = pipeline(example_img)

    left_fit, right_fit, left_lane_inds, right_lane_inds, visualization_data = sliding_window_polyfit(example_img_bin)

    h = example_img.shape[0]
    left_fit_x_int = left_fit[0] * h ** 2 + left_fit[1] * h + left_fit[2]
    right_fit_x_int = right_fit[0] * h ** 2 + right_fit[1] * h + right_fit[2]
    # print('fit x-intercepts:', left_fit_x_int, right_fit_x_int)

    rectangles = visualization_data[0]
    histogram = visualization_data[1]

    plt.plot(histogram)
    plt.xlim(0, 1280)
    file_name = "./output_images/polyfit_result_histogram.png"
    plt.savefig(file_name, bbox_inches='tight')
    plt.close('all')

    # Create an output image to draw on and  visualize the result
    out_img = np.uint8(np.dstack((example_img_bin, example_img_bin, example_img_bin)) * 255)
    # Generate x and y values for plotting
    ploty = np.linspace(0, example_img_bin.shape[0] - 1, example_img_bin.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    for rect in rectangles:
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (rect[2], rect[0]), (rect[3], rect[1]), (0, 255, 0), 2)
        cv2.rectangle(out_img, (rect[4], rect[0]), (rect[5], rect[1]), (0, 255, 0), 2)
        # Identify the x and y positions of all nonzero pixels in the image
    nonzero = example_img_bin.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [100, 200, 255]
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)

    file_name = "./output_images/polyfit_result.png"
    plt.savefig(file_name, bbox_inches='tight')
    plt.close('all')

    ############## POLYFIT USING PREVIOUS ####################

    # visualize the result on example image
    example_img2 = cv2.imread('./test_images/test2.jpg')
    example_img2 = cv2.cvtColor(example_img2, cv2.COLOR_BGR2RGB)
    exampleImg2_bin, minv = pipeline(example_img2)
    margin = 80

    left_fit2, right_fit2, left_lane_inds2, right_lane_inds2 = polyfit_using_previous_fit(exampleImg2_bin, left_fit,
                                                                                          right_fit)

    # Generate x and y values for plotting
    ploty = np.linspace(0, exampleImg2_bin.shape[0] - 1, exampleImg2_bin.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    left_fitx2 = left_fit2[0] * ploty ** 2 + left_fit2[1] * ploty + left_fit2[2]
    right_fitx2 = right_fit2[0] * ploty ** 2 + right_fit2[1] * ploty + right_fit2[2]

    # Create an image to draw on and an image to show the selection window
    out_img = np.uint8(np.dstack((exampleImg2_bin, exampleImg2_bin, exampleImg2_bin)) * 255)
    window_img = np.zeros_like(out_img)

    # Color in left and right line pixels
    nonzero = exampleImg2_bin.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    out_img[nonzeroy[left_lane_inds2], nonzerox[left_lane_inds2]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds2], nonzerox[right_lane_inds2]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area (OLD FIT)
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    plt.imshow(result)
    plt.plot(left_fitx2, ploty, color='yellow')
    plt.plot(right_fitx2, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    file_name = "./output_images/polyfit_result_using_prev.png"
    plt.savefig(file_name, bbox_inches='tight')
    plt.close('all')

    ############## CURVATURE RESULTS ####################
    rad_l, rad_r, d_center = curvature_calculator(example_img_bin, left_fit, right_fit, left_lane_inds,
                                                  right_lane_inds)

    print('Radius of curvature for test2.jpg:', rad_l, 'm,', rad_r, 'm')
    print('Distance from lane center for test2.jpg:', d_center, 'm')

    ################ DRAW LANE ################
    exampleImg_out1 = draw_lane(example_img, example_img_bin, left_fit, right_fit, minv, (rad_l + rad_r) / 2, d_center)
    plt.imshow(exampleImg_out1)
    file_name = "./output_images/polyfit_result_with_lanes.png"
    plt.savefig(file_name, bbox_inches='tight')
