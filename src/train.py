from ultralytics import YOLO
from utils import clear_gpu

# Configure this for your own dataset
DATA_PATH = "C:/Weapon-detection-1/data.yaml" 
MODEL_VARIANT = "yolo26n.pt"

# Optuna's best params
BEST_PARAMS = {
    'lr0': 0.0006470951009881973, 
    'weight_decay': 7.127127820764751e-05, 
    'momentum': 0.8576810045292452, 
    'warmup_epochs': 2.8531705580385065, 
    'box': 7.6353443525209705, 
    'cls': 0.29836457704140795, 
    'dfl': 1.7355025410937515
}

def run_training():
    clear_gpu()
    model = YOLO(MODEL_VARIANT)
    
    results = model.train(
        data=DATA_PATH,
        epochs=100,
        imgsz=640,
        batch=16,
        workers=2,
        device=0,
        cache=False,
        name="weapon_detection_final",
        **BEST_PARAMS
    )
    return results

if __name__ == "__main__":
    run_training()