import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/media/cc/YOLOv11-RGBT/ultralytics/cfg/models/11/yolov11n-gray.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data=R'/media/cc/YOLOv11-RGBT/ultralytics/cfg/datasets/coco8_hand.yaml',
                cache=False,
                multi_scale=True,
                imgsz=320,
                epochs=1000,
                save_period=100,
                batch=128,
                close_mosaic=10,
                workers=32,
                device=[0],
                # optimizer='SGD',  # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                use_simotm="Gray16bit",
                channels=1,
                project='HAND-1C',
                name='HAND-yolov11n-GRAY',
                )
