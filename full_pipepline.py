import numpy as np

from pipeline import pipeline
import line
from sliding_window_polyfit import sliding_window_polyfit, polyfit_using_previous_fit, curvature_calculator
from lane_drawer import draw_lane
from moviepy.editor import VideoFileClip


class final_pieline_video:
    def __init__(self):
        self.l_line = line.Line()
        self.r_line = line.Line()

    def pipeline(self, img):
        new_img = np.copy(img)

        img_bin, minv = pipeline(new_img)

        if not self.l_line.detected or not self.r_line.detected:
            left, r_fit, l_lane_inds, r_lane_inds, _ = sliding_window_polyfit(img_bin)
        else:
            left, r_fit, l_lane_inds, r_lane_inds = polyfit_using_previous_fit(img_bin, self.l_line.best_fit,
                                                                               self.r_line.best_fit)

        self.l_line.add_fit(left, l_lane_inds)
        self.r_line.add_fit(r_fit, r_lane_inds)

        # draw the current best fit if it exists
        if self.l_line.best_fit is not None and self.r_line.best_fit is not None:
            rad_l, rad_r, d_center = curvature_calculator(img_bin, self.l_line.best_fit, self.r_line.best_fit,
                                                          l_lane_inds, r_lane_inds)
            img_out = draw_lane(new_img, img_bin, self.l_line.best_fit, self.r_line.best_fit, minv,
                                (rad_l + rad_r) / 2.0, d_center)
        else:
            img_out = new_img

        return img_out


if __name__ == "__main__":
    video_output1 = 'project_video_output.mp4'
    video_input1 = VideoFileClip('project_video.mp4')
    fp = final_pieline_video()
    processed_video = video_input1.fl_image(fp.pipeline)
    processed_video.write_videofile(video_output1, audio=False)
    #
    # video_output1 = 'project_challenge_video_output.mp4'
    # video_input1 = VideoFileClip('challenge_video.mp4')
    # fp = final_pieline_video()
    # processed_video = video_input1.fl_image(fp.pipeline)
    # processed_video.write_videofile(video_output1, audio=False)

    # video_output1 = 'project_harder_challenge_video_output.mp4'
    # video_input1 = VideoFileClip('harder_challenge_video.mp4')
    # fp = final_pieline_video()
    # processed_video = video_input1.fl_image(fp.pipeline)
    # processed_video.write_videofile(video_output1, audio=False)
