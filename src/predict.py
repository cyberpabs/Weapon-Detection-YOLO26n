import sys
from ultralytics import YOLO

def infer(source_path, model_path='models/best.pt'):
    model = YOLO(model_path)
    results = model.predict(source=source_path, save=True, conf=0.5)
    print(f"Results saved on: {results[0].save_dir}")

if __name__ == "__main__":
    # Ús: python src/predict.py imatge.jpg
    if len(sys.argv) > 1:
        infer(sys.argv[1])
    else:
        print("Error: Indicate the path to an image or video.")