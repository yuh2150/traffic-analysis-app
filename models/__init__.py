from .yolov5_wrapper import YOLOv5Wrapper, BaseTrafficDetector, UA_DETRAC_CLASSES, NUM_UA_DETRAC_CLASSES
from .detr_wrapper import DETRWrapper
from .factory import ModelFactory

def get_model(model_name: str, weights_path: str = "", device: str = "cpu"):
    return ModelFactory.load(model_name, weights_path, device)
