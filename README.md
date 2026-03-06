# Weapon Detection System (YOLO26n + Optuna)

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Demo-FF4B4B.svg)](https://streamlit.io/)

A real-time weapon identification system designed for security environments. This project features a **YOLO26n** architecture with both default and **Bayesian Optimization (Optuna)** to achieve high precision with low latency.

## 🚀 Key Features
* **Multi-Format Analysis:** Support for Images (Grid View), Video files, and Live Webcam.
* **Interactive Dashboard:** Built with Streamlit for easy model comparison and testing.

## 📊 Performance Comparison
| Model | mAP@50 | mAP@50-95 | Latency (ms) |
| :--- | :---: | :---: | :---: |
| YOLO26n (Base) | 0.9318 | 0.5779      | ~4ms |
| YOLO26n (Tuned) | 0.9207 | 0.5712      | ~4ms |

## 🛠️ Installation & Usage
1. Clone the repo: `git clone https://github.com/cyberpabs/Weapon-Detection-YOLO26-Optuna`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the demo: `streamlit run app.py`

## 🧪 Methodology
The model was trained on a dataset of firearms and knives. Hyperparameters like `lr0`, `momentum`, and `box loss` were fine-tuned over 100 iterations using a Bayesian Search strategy.
