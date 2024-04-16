import pytz
import cv2
import json
from collections import deque
from datetime import datetime

# Load JSON config
with open('examples/config.json', 'r') as file:
    config = json.load(file)
class ObjectCounter:
    def __init__(self):
        self.line = config['object_counter_settings']['line_coordinates']
        self.data_deque = {}  # Tracks object centers (deque for each object ID).
        self.object_counter = {}  # Counting objects moving in one direction
        self.object_counter1 = {}  # Counting objects moving in the opposite direction
        self.crossing_time_in = {}
        self.crossing_time_out = {}

        # Initialize dictionaries to store the timestamp of crossing events
        crossing_time_in = {}  # For 'West' direction crossings
        crossing_time_out = {}  # For 'East' direction crossings

        self.india_time_zone = pytz.timezone(config['object_counter_settings']['timezone'])

    def ccw(self, A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def intersect(self, A, B, C, D):
        return self.ccw(A, C, D) != self.ccw(B, C, D) and self.ccw(A, B, C) != self.ccw(A, B, D)

    def get_direction(self, point1, point2):
        direction_str = ""

        # calculate y axis direction
        if point1[1] > point2[1]:
            direction_str += "South"
        elif point1[1] < point2[1]:
            direction_str += "North"
        else:
            direction_str += ""

        # calculate x axis direction
        if point1[0] > point2[0]:
            direction_str += "East"
        elif point1[0] < point2[0]:
            direction_str += "West"
        else:
            direction_str += ""

        return direction_str

    def update_counters(self, pred_boxes, line):
        for d in reversed(pred_boxes):
            xyxy = d.xyxy
            x1, y1, x2, y2 = xyxy[0][0], xyxy[0][1], xyxy[0][2], xyxy[0][3]
            center = (int((x2 + x1) / 2), int((y2 + y1) / 2))
            obj_id = None if d.id is None else int(d.id.item())
            obj_name = 'person'

            if obj_id not in self.data_deque:
                self.data_deque[obj_id] = deque(maxlen=64)

            self.data_deque[obj_id].appendleft(center)
            if len(self.data_deque[obj_id]) >= 2:
                direction = self.get_direction(self.data_deque[obj_id][0], self.data_deque[obj_id][1])
                if self.intersect(self.data_deque[obj_id][0], self.data_deque[obj_id][1], line[0], line[1]):
                    if "West" in direction:
                        self.object_counter[obj_name] = self.object_counter.get(obj_name, 0) + 1
                        time_in = datetime.now(self.india_time_zone)
                        self.crossing_time_in[obj_id] = time_in
                        self.crossing_time_out.pop(obj_id, None)
                    elif "East" in direction:
                        self.object_counter1[obj_name] = self.object_counter1.get(obj_name, 0) + 1
                        time_out = datetime.now(self.india_time_zone)
                        self.crossing_time_out[obj_id] = time_out

    def draw_counters_on_image(self,plotted_img, counter):
        # Drawing a line and rectangles on the image
        cv2.line(plotted_img, self.line[0], self.line[1], (255, 0, 0), 3)
        cv2.rectangle(plotted_img, (0, 0), (450, 438), (255, 0, 0), 3)
        cv2.rectangle(plotted_img, (0, 440), (1000, 500), (0, 0, 0), -1)

        # Initialize total counters
        total_in = 0
        total_out = 0

        # Drawing text for objects moving out
        for idx, (key, value) in enumerate(self.object_counter1.items()):
            cnt_str1 = f"{key}-out: {value}"
            total_out += value
            cv2.putText(plotted_img, cnt_str1, (500, 465 - (idx * 20)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2, 2)

        # Drawing text for objects moving in
        for idx, (key, value) in enumerate(self.object_counter.items()):
            cnt_str = f"{key}-in: {value}"
            total_in += value
            cv2.putText(plotted_img, cnt_str, (20, 465 - (idx * 20)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2, 2)

        # Calculating and drawing the total count
        total = total_in - total_out
        total_str = f"total: {total}"
        cv2.putText(plotted_img, total_str, (300, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, 2)

        return plotted_img