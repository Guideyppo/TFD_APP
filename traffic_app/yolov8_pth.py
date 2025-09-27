import torch
from ultralytics.nn.tasks import DetectionModel

def load_yolov8_pth(pth_path, num_classes=3, model_size="n"):
    # เลือก config yaml ที่ตรงกับตอนเทรน
    yaml_path = f"yolov8{model_size}.yaml"   # เช่น yolov8n.yaml, yolov8s.yaml
    
    # สร้างโมเดลใหม่
    model = DetectionModel(yaml_path, nc=num_classes, verbose=False)

    # โหลด weight จากไฟล์ .pth (state_dict)
    state_dict = torch.load(pth_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model