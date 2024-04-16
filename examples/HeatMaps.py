
from collections import defaultdict

import cv2
import numpy as np

from ultralytics.utils.checks import check_imshow, check_requirements
from ultralytics.utils.plotting import Annotator
from ultralytics.solutions.heatmap import Heatmap

check_requirements('shapely>=2.0.0')

from shapely.geometry import LineString, Point, Polygon
class newHeatmap(Heatmap):

    def __init__(self):
        """Initializes the heatmap class with default values for Visual, Image, track, count and heatmap parameters."""

        # Visual information
        self.annotator = None
        self.view_img = False
        self.shape = 'circle'

        # Image information
        self.imw = None
        self.imh = None
        self.im0 = None

        # Heatmap colormap and heatmap np array
        self.colormap = None
        self.heatmap = None
        self.heatmap_alpha = 0.5

        # Predict/track information
        self.boxes = None
        self.track_ids = None
        self.clss = None
        self.track_history = defaultdict(list)

        # Region & Line Information
        self.count_reg_pts = None
        self.counting_region = None
        self.line_dist_thresh = 15
        self.region_thickness = 5
        self.region_color = (255, 0, 255)

        # Object Counting Information
        self.in_counts = 0
        self.out_counts = 0
        self.counting_list = []
        self.count_txt_thickness = 0
        self.count_txt_color = (0, 0, 0)
        self.count_color = (255, 255, 255)

        # Decay factor
        self.decay_factor = 0.99

        self.track_thickness = 2
        self.track_color = (0, 255, 0)

        # Check if environment support imshow
        self.env_check = check_imshow(warn=True)

    def set_args(self,
                 imw,
                 imh,
                 colormap=cv2.COLORMAP_JET,
                 heatmap_alpha=0.5,
                 view_img=False,
                 count_reg_pts=None,
                 count_txt_thickness=2,
                 count_txt_color=(0, 0, 0),
                 count_color=(255, 255, 255),
                 count_reg_color=(255, 0, 255),
                 region_thickness=5,
                 line_dist_thresh=15,
                 decay_factor=0.99,
                 shape='circle'):
        """
        Configures the heatmap colormap, width, height and display parameters.

        Args:
            colormap (cv2.COLORMAP): The colormap to be set.
            imw (int): The width of the frame.
            imh (int): The height of the frame.
            heatmap_alpha (float): alpha value for heatmap display
            view_img (bool): Flag indicating frame display
            count_reg_pts (list): Object counting region points
            count_txt_thickness (int): Text thickness for object counting display
            count_txt_color (RGB color): count text color value
            count_color (RGB color): count text background color value
            count_reg_color (RGB color): Color of object counting region
            region_thickness (int): Object counting Region thickness
            line_dist_thresh (int): Euclidean Distance threshold for line counter
            decay_factor (float): value for removing heatmap area after object passed
            shape (str): Heatmap shape, rect or circle shape supported
        """
        self.imw = imw
        self.imh = imh
        self.heatmap_alpha = heatmap_alpha
        self.view_img = view_img
        self.colormap = colormap

        # Region and line selection
        if count_reg_pts is not None:

            if len(count_reg_pts) == 2:
                print('Line Counter Initiated.')
                self.count_reg_pts = count_reg_pts
                self.counting_region = LineString(count_reg_pts)

            elif len(count_reg_pts) == 4:
                print('Region Counter Initiated.')
                self.count_reg_pts = count_reg_pts
                self.counting_region = Polygon(self.count_reg_pts)

            else:
                print('Region or line points Invalid, 2 or 4 points supported')
                print('Using Line Counter Now')
                self.counting_region = Polygon([(20, 400), (1260, 400)])  # dummy points

        # Heatmap new frame
        self.heatmap = np.zeros((int(self.imw), int(self.imh)), dtype=np.float32)

        self.count_txt_thickness = count_txt_thickness
        self.count_txt_color = count_txt_color
        self.count_color = count_color
        self.region_color = count_reg_color
        self.region_thickness = region_thickness
        self.decay_factor = decay_factor
        self.line_dist_thresh = line_dist_thresh
        self.shape = shape

        # shape of heatmap, if not selected
        if self.shape not in ['circle', 'rect']:
            print("Unknown shape value provided, 'circle' & 'rect' supported")
            print('Using Circular shape now')
            self.shape = 'circle'
    def generate_heatmap(self, im0, tracks):

        """
        Generate heatmap based on tracking data.

        Args:
            im0 (nd array): Image
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self.im0 = im0
        if tracks[0].boxes.id is None:
            return self.im0

        self.heatmap *= self.decay_factor  # decay factor
        self.extract_results(tracks)
        self.annotator = Annotator(self.im0, self.count_txt_thickness, None)



        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            track_line = self.track_history[track_id]
            track_line.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
            if len(track_line) > 30:
                track_line.pop(0)

            # Draw track trails
            self.annotator.draw_centroid_and_tracks(track_line,
                                                    color=self.track_color,
                                                    track_thickness=self.track_thickness)

            if self.shape == 'circle':
                center = (int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2))
                radius = min(int(box[2]) - int(box[0]), int(box[3]) - int(box[1])) // 2

                y, x = np.ogrid[0:self.heatmap.shape[0], 0:self.heatmap.shape[1]]
                mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2

                self.heatmap[int(box[1]):int(box[3]), int(box[0]):int(box[2])] += \
                    (2 * mask[int(box[1]):int(box[3]), int(box[0]):int(box[2])])

            else:
                self.heatmap[int(box[1]):int(box[3]), int(box[0]):int(box[2])] += 2

        # Normalize, apply colormap to heatmap and combine with original image
        heatmap_normalized = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), self.colormap)


        im0_with_heatmap = cv2.addWeighted(self.im0, 1 - self.heatmap_alpha, heatmap_colored, self.heatmap_alpha, 0)


        return im0_with_heatmap