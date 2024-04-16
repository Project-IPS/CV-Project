
from pathlib import Path
import numpy as np
import websocket
import json
import cv2
import time
import base64

from io import BytesIO
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import LOGGER, colorstr, ops
from ultralytics.utils.torch_utils import smart_inference_mode
from ultralytics.engine.predictor import BasePredictor
from ultralytics.solutions import heatmap
from examples.HeatMaps import newHeatmap
from ultralytics.engine.results import Results
from datetime import datetime

from examples.counter import ObjectCounter
from examples.TacticalMap import TacticalMap
from examples.WebSocket import WebSocket

# Create global instances
counter = ObjectCounter()
websocket = WebSocket()
tactical_map = TacticalMap()

class CustomPredictor(BasePredictor):

    ## Subclass adds counter and other features to ultralytics's BasePredictor
    def init(self):
        global counter
        global websocket
        global tactical_map

        self.line = [(450, 0), (450, 600)]
        self.cnt_str = None
        self.cnt_str1 = None
        self.inn = 0
        self.out = 0
        self.last_sent_inn = 0
        self.last_sent_out = 0
        self.entry_sent = set()  # To keep track of which entry data has been sent
        self.frame_count = 0
        ## init heatmap
        self.heatmap_obj = newHeatmap()
        self.heatmap_obj.set_args(colormap=cv2.COLORMAP_PARULA,
                                  imw=480,
                                  imh=640,
                                  view_img=False,
                                  shape="circle")
    @smart_inference_mode()
    def stream_inference(self, source=None, model=None, *args, **kwargs):

        self.init()

        """Streams real-time inference on camera feed and saves results to file."""
        if self.args.verbose:
            LOGGER.info('')

        # Setup model
        if not self.model:
            self.setup_model(model)

        with (self._lock):  # for thread-safe inference

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
                print(type(self.results[0]))
                self.run_callbacks('on_predict_postprocess_end')
                print(type(self.results[0]))
                # Visualize, save, write results
                n = len(im0s)
                for i in range(n):
                    self.seen += 1
                    self.results[i].speed = {
                        'preprocess': profilers[0].dt * 1E3 / n,
                        'inference': profilers[1].dt * 1E3 / n,
                        'postprocess': profilers[2].dt * 1E3 / n}
                    p, im0 = path[i], None if self.source_type.tensor else im0s[i].copy()
                    p = Path(p)

                    #################################counting############################################

                    pred_boxes = self.results[i].boxes
                    counter.update_counters(pred_boxes, self.line)

                    ####################################################################################################

                    #Add bounding boxes to the image
                    if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                        s += self.write_results(i, self.results, (p, im, im0))

                    # Draw counters inference on the image
                    self.plotted_img = counter.draw_counters_on_image(self.plotted_img, counter)

                    # heatmap infernece
                    self.plotted_img = self.heatmap_obj.generate_heatmap(self.plotted_img, self.results)

                    #############################Tactical map############################################

                    # tactical_map = TacticalMap()
                    tactical_map.pred_dst_pts = {}
                    tac_map = tactical_map.update_tactical_map(pred_boxes)

                    # Define the desired width and height for the resized image
                    new_width = 800
                    new_height = 600

                    # Resize the image
                    resized_image = cv2.resize(self.plotted_img, (new_width, new_height))
                    tac_map = cv2.resize(tac_map, (new_width, new_height))  # Resize tactical map
                    final_img = cv2.hconcat((resized_image, tac_map))

                    cv2.imshow("Lab tracking",final_img)
                    cv2.waitKey(500 if self.batch[3].startswith('image') else 1)  # 1 millisecond

                    #####################################websocket#######################################################

                    websocket.inn = self.inn
                    websocket.out = self.out
                    websocket.crossing_time_in = counter.crossing_time_in
                    websocket.crossing_time_out = counter.crossing_time_out
                    websocket.pred_dst_pts = tactical_map.pred_dst_pts

                    websocket.Response()    #send data to websocket server
                    ####################################################################################################


                    # Remove the IDs that are common in both dictionaries i.e they entered and left
                    common_keys = set(counter.crossing_time_in.keys()) & set(counter.crossing_time_out.keys())
                    for key in common_keys:
                        counter.crossing_time_in.pop(key)
                        counter.crossing_time_out.pop(key)


                    #######################################################

                    if self.args.save or self.args.save_txt:
                        self.results[i].save_dir = self.save_dir.__str__()
                    # if self.args.show and self.plotted_img is not None:
                    #     self.show(p)
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

    def run_callbacks(self, event: str):
        """Runs all registered callbacks for a specific event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def add_callback(self, event: str, func):
        """Add callback."""
        self.callbacks[event].append(func)
