import numpy as np
import cv2
import json

# Load JSON config
with open('examples/config.json', 'r') as file:
    config = json.load(file)
class TacticalMap:

    def __init__(self):
        self.detected_labels_src_pts = np.array(config['tactical_map_settings']['source_points'])
        self.detected_labels_dst_pts = np.array(config['tactical_map_settings']['destination_points'])
        self.h, _ = cv2.findHomography(self.detected_labels_src_pts, self.detected_labels_dst_pts)  ## Calculate homography matrix
        self.tac_map = cv2.imread(config['tactical_map_settings']['image_path'])
        self.pred_dst_pts = {}

    def annotate_tactical_map(self):

        tac_map_copy = self.tac_map.copy()

        # Loop over detected people and add tactical map annotations
        for _, (person_id, person_pos) in enumerate(self.pred_dst_pts.items()):

            # Add tactical map people position color-coded annotation if more than 3 keypoints are detected
            if len(self.detected_labels_src_pts) > 3:
                tac_map_copy = cv2.circle(tac_map_copy,                 ## Add tactical map people position color-coded annotation
                                          (int(person_pos[0]), int(person_pos[1])),
                                          radius=20, color=(255, 0, 0), thickness=-1)

                cv2.putText(tac_map_copy, str(int(person_id)), (int(person_pos[0]) - 8, int(person_pos[1]) + 8),  ## Add player id
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Combine annotated frame and tactical map in one image with colored border separation
        border_color = [255, 255, 255]  # Set border color (BGR)
        tac_map_copy = cv2.copyMakeBorder(tac_map_copy, 70, 50, 10, 10, cv2.BORDER_CONSTANT,value=border_color)  # Add borders to tactical map
        return tac_map_copy


    def update_tactical_map(self, pred_boxes):
        bboxes_id = None if pred_boxes.id is None else pred_boxes.id.cpu().numpy()
        bboxes_p_c = pred_boxes.xywh.cpu().numpy()

        detected_ppos_src_pts = bboxes_p_c[:, :2] + np.array([[0] * bboxes_p_c.shape[0], bboxes_p_c[:, 3] / 2]).transpose()


        for idx, pt in enumerate(detected_ppos_src_pts):
            pt = np.append(np.array(pt), np.array([1]), axis=0)
            dest_point = np.matmul(self.h, np.transpose(pt))
            dest_point = dest_point / dest_point[2]
            person_id = bboxes_id[idx]
            self.pred_dst_pts[person_id] = list(np.transpose(dest_point)[:2])

        tac_map_copy = self.annotate_tactical_map()
        return tac_map_copy