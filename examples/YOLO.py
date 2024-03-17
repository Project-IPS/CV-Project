# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.models.yolo.model import YOLO
from ultralytics.models import yolo  # noqa
from examples.DetectionPredictor import newDetectionPredictor
from ultralytics.nn.tasks import DetectionModel



class newYOLO(YOLO):
    """subclass of YOLO, changes the mapping of predictor to newDetectionPredictor
                    (which is a subclass of CustomPredictor(which is subclass of BasePredictor))"""

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            'detect': {
                'model': DetectionModel,
                'trainer': yolo.detect.DetectionTrainer,
                'validator': yolo.detect.DetectionValidator,
                'predictor': newDetectionPredictor, },
        }
