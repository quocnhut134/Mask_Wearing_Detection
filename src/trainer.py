from ultralytics import YOLO
from src import config

class MaskTrainer:
    def __init__(self):
        self.model = YOLO(config.model_name)
    
    def train(self):
        results = self.model.train(
            data=config.data_yaml_path,
            epochs=config.epochs,
            imgsz=config.image_size,
            batch=config.batch_size,
            name=config.project_name,
            device=config.device,
            amp=False,       
            workers=config.workers 
        )
        
        return results

    def validate(self):
        metrics = self.model.val()
        print(f"mAP50: {metrics.box.map50}")
        print(f"mAP50-95: {metrics.box.map}")
        return metrics