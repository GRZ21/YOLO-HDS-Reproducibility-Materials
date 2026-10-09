from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("configs/yolo-hds.yaml", task="detect")
    yolo_model = model.model  # DetectionModel

    model.train(data="configs/melon.yaml",
                batch=64,
                epochs=100,
                workers=8,
                imgsz=640,
                device=0,
                project="./runs",
                resume=False,
                pretrained=False,
                close_mosaic=10,
                amp=True,
                optimizer="AdamW",   
                seed=0,              
                deterministic=True,
               )