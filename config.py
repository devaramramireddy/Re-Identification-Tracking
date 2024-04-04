# tracker_config.py
from pydantic import BaseModel
from typing import Optional

class Config(BaseModel):
    yolo_model_path: Optional[str] = "./models/yolov5su.pt"
    reid_model_name: Optional[str] = "osnet_ain_x1_0"
    reid_model_path: Optional[str] = "./models/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth"
    input_type: Optional[str] = "RGB"
    video_path: Optional[str] = "./data/Person_counting_assignment.mp4"
    similarity_threshold_yolo: Optional[float] = 0.8
    similarity_threshold_reid: Optional[float] = 0.7
    tracking_id: Optional[int] = 1
    output_path: Optional[str] = "./Results"
    preprocessing: Optional[dict] = {
        "color_normalization": False,
        "background_subtraction": False,
        "noise_reduction": True,
        "histogram_equalization": True
    }
