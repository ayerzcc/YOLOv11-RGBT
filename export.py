import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# onnx onnxsim onnxruntime onnxruntime-gpu

# 导出参数官方详解链接：https://docs.ultralytics.com/modes/export/#usage-examples

if __name__ == '__main__':
    model = YOLO('./best.pt')
    model.export(format='onnx', 
                 imgsz=320, 
                 opset=14,
                 channels=1,
                 data='/media/cc/ultralytics_hand_det/ultralytics/cfg/datasets/coco8_hand.yaml')

    #  tensorrt   若无特殊考虑，务必把动态尺寸设置为False  # https://docs.ultralytics.com/zh/integrations/tensorrt/#usage
    # # Export the model to TensorRT format
    # model.export(format="engine", channels=6)  # creates 'yolo11n.engine'
    #
    # # Load the exported TensorRT model
    # tensorrt_model = YOLO("yolo11n.engine")
    #
    # # Run inference
    # results = tensorrt_model("https://ultralytics.com/images/bus.jpg")