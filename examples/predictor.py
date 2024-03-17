
from pathlib import Path

import cv2


from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import LOGGER, colorstr, ops
from ultralytics.utils.torch_utils import smart_inference_mode
from collections import deque
from ultralytics.engine.predictor import BasePredictor



class CustomPredictor(BasePredictor):

    ## Subclass adds counter feature to BasePredictor

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

    @smart_inference_mode()
    def stream_inference(self, source=None, model=None, *args, **kwargs):
        data_deque = {}  # Tracking object centers
        object_counter = {}  # Counting objects moving in one direction
        object_counter1 = {}  # Counting objects moving in the opposite direction
        line = [(450, 0), (450, 600)]  # Define your counting line
        self.cnt_str=None
        self.cnt_str1=None
        self.inn=0
        self.out=0

        """Streams real-time inference on camera feed and saves results to file."""
        if self.args.verbose:
            LOGGER.info('')

        # Setup model
        if not self.model:
            self.setup_model(model)

        with self._lock:  # for thread-safe inference
            # Setup source every time predict is called
            self.setup_source(source if source is not None else self.args.source)

            # Check if save_dir/ label file exists
            if self.args.save or self.args.save_txt:
                (self.save_dir / 'labels' if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

            # Warmup model
            if not self.done_warmup:
                self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
                self.done_warmup = True

            self.seen, self.windows, self.batch, profilers = 0, [], None, (ops.Profile(), ops.Profile(), ops.Profile())
            self.run_callbacks('on_predict_start')

            for batch in self.dataset:
                self.run_callbacks('on_predict_batch_start')
                self.batch = batch
                path, im0s, vid_cap, s = batch  ####

                # Preprocess
                with profilers[0]:
                    im = self.preprocess(im0s)

                # Inference          ###############
                with profilers[1]:
                    preds = self.inference(im, *args, **kwargs)

                # Postprocess
                with profilers[2]:
                    if isinstance(self.model, AutoBackend):
                        self.results = self.postprocess(preds, im, im0s)
                    else:
                        self.results = self.model.postprocess(path, preds, im, im0s)

                self.run_callbacks('on_predict_postprocess_end')
                # Visualize, save, write results
                n = len(im0s)
                for i in range(n):
                    self.seen += 1
                    self.results[i].speed = {
                        'preprocess': profilers[0].dt * 1E3 / n,
                        'inference': profilers[1].dt * 1E3 / n,
                        'postprocess': profilers[2].dt * 1E3 / n}
                    p, im0 = path[i], None if self.source_type.tensor else im0s[i].copy()
                    print(p)
                    p = Path(p)
                    #################################
                    self.count = len(self.results[i].boxes) ## number of objects in frame
                    
                                                ##################Line Based Counter##############

                    pred_boxes = self.results[i].boxes
                    for d in reversed(pred_boxes):
                        xyxy = d.xyxy
                        x1 = xyxy[0][0]
                        y1 = xyxy[0][1]
                        x2 = xyxy[0][2]
                        y2 = xyxy[0][3]
                        center = (int((x2 + x1) / 2), int((y2 + y2) / 2))
                        id = None if d.id is None else int(d.id.item())
                        obj_name = 'person'
                        # create new buffer for new object
                        if id not in data_deque:
                            data_deque[id] = deque(maxlen=64)

                        data_deque[id].appendleft(center)
                        if len(data_deque[id]) >= 2:
                            direction = self.get_direction(data_deque[id][0], data_deque[id][1])
                            if self.intersect(data_deque[id][0], data_deque[id][1], line[0], line[1]):

                                # if "South" in direction:
                                #     if obj_name not in object_counter:
                                #         object_counter[obj_name] = 1
                                #     else:
                                #         object_counter[obj_name] += 1
                                # elif "North" in direction:
                                #     if obj_name not in object_counter1:
                                #         object_counter1[obj_name] = 1
                                #     else:
                                #         object_counter1[obj_name] += 1
                                if "West" in direction:
                                    if obj_name not in object_counter:
                                        object_counter[obj_name] = 1
                                    else:
                                        object_counter[obj_name] += 1
                                elif "East" in direction:
                                    if obj_name not in object_counter1:
                                        object_counter1[obj_name] = 1
                                    else:
                                        object_counter1[obj_name] += 1
                    ##
                    if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                        s += self.write_results(i, self.results, (p, im, im0))
                    ##

                    line = [(450, 0), (450, 600)]
                    cv2.line(self.plotted_img, line[0], line[1], (255, 0, 0), 3)
                    cv2.rectangle(self.plotted_img, (0, 0), (450, 438), (255, 0, 0), 3)
                    cv2.rectangle(self.plotted_img, (0, 440), (1000, 500), (0, 0, 0), -1)
                    for idx, (key, value) in enumerate(
                            object_counter1.items()):  ## using for loop incase we have more than 1 class
                        self.cnt_str1 = str(key) + "-out" + ":" + str(value)
                        self.out = value
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        bottomLeftCornerOfText = (500, 465)
                        fontScale = 0.6
                        fontColor = (255, 255, 255)
                        thickness = 2
                        lineType = 2

                        cv2.putText(self.plotted_img, self.cnt_str1, bottomLeftCornerOfText, font, fontScale, fontColor,
                                    thickness, lineType)

                    for idx, (key, value) in enumerate(object_counter.items()):
                        self.cnt_str = str(key) + "-in" ":" + str(value)
                        self.inn = value
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        bottomLeftCornerOfText = (20, 465)
                        fontScale = 0.6
                        fontColor = (255, 255, 255)
                        thickness = 2
                        lineType = 2

                        cv2.putText(self.plotted_img, self.cnt_str, bottomLeftCornerOfText, font, fontScale, fontColor,
                                    thickness, lineType)

                    Total = self.inn - self.out
                    Total_str = "total: " + str(Total)
                    cv2.putText(self.plotted_img, Total_str, (300, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
                                2, 2)

                    if self.args.save or self.args.save_txt:
                        self.results[i].save_dir = self.save_dir.__str__()
                    if self.args.show and self.plotted_img is not None:
                        self.show(p)
                    if self.args.save and self.plotted_img is not None:
                        self.save_preds(vid_cap, i, str(self.save_dir / p.name))

                self.run_callbacks('on_predict_batch_end')

                yield from self.results

                # Print time (inference-only)
                if self.args.verbose:
                    LOGGER.info(f'{s}{profilers[1].dt * 1E3:.1f}ms')

            # Release assets
        if isinstance(self.vid_writer[-1], cv2.VideoWriter):
            self.vid_writer[-1].release()  # release final video writer

            # Print results
        if self.args.verbose and self.seen:
            t = tuple(x.t / self.seen * 1E3 for x in profilers)  # speeds per image
            LOGGER.info(f'Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape '
                        f'{(1, 3, *im.shape[2:])}' % t)
        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob('labels/*.txt')))  # number of labels
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")

            # Finalize and clean up if necessary
            self.run_callbacks('on_predict_end')
