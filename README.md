# YOLOv5 and YOLOv8 Performance Comparison for PPE Detection

This repository contains the implementation and evaluation of object detection models using YOLOv5 and YOLOv8 to detect Personal Protective Equipment (PPE) components such as Helmets, IFR Suits, and Boots. The project explores model performance under different training configurations and highlights key insights based on metrics like Precision, Recall, and mAP.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Configurations](#model-configurations)
- [Results](#results)
- [Key Insights](#key-insights)
- [Usage](#usage)
- [References](#references)

## Introduction

This project aims to evaluate the performance of YOLOv5 and YOLOv8 on a PPE detection dataset. Various training configurations, including epoch increases and hyperparameter tuning, were tested to optimize the models and improve detection accuracy.

## Dataset

The dataset used for this project contains labeled images for three PPE components:
- **Helmet**
- **IFR Suit**
- **Boots**

The dataset was split into:
- **Training Set**
- **Validation Set**
- **Testing Set**

## Model Configurations

### YOLOv5
1. **Baseline (50 Epochs):**
   - Standard training with no hyperparameter tuning.
2. **Enhanced (100 Epochs & Hypertuning):**
   - Training extended to 100 epochs with optimized hyperparameters.

### YOLOv8
1. **Baseline (50 Epochs):**
   - Evaluated using YOLOv8’s default configuration for comparison with YOLOv5.

## Results

| Metric           | YOLOv5 Baseline (50 Epochs) | YOLOv5 (100 Epochs & Hypertuning) | YOLOv8 Baseline (50 Epochs) |
|-------------------|-----------------------------|------------------------------------|-----------------------------|
| **Precision**     | 0.56689                    | 0.46967                            | 0.4584                     |
| **Recall**        | 0.32576                    | 0.48601                            | 0.6312                     |
| **mAP@0.5**       | 0.17733                    | 0.43785                            | 0.4808                     |
| **mAP@0.5:0.95**  | 0.06062                    | 0.17525                            | 0.2360                     |

### Summary
- **YOLOv5 Baseline:** Showed good precision but lower recall and mAP.
- **YOLOv5 with Improvements:** Enhanced recall and mAP significantly with more epochs and tuning.
- **YOLOv8 Baseline:** Outperformed YOLOv5 in recall and mAP, even in the baseline configuration.

## Key Insights

1. Increasing epochs and applying hyperparameter tuning in YOLOv5 improved recall and mAP scores.
2. YOLOv8 demonstrated superior performance in terms of recall and overall detection accuracy, achieving the highest mAP scores across configurations.
3. YOLOv5 has a slight advantage in precision when using fewer epochs, but this decreases when epochs increase.

## Usage

### Requirements
- Python 3.10+
- PyTorch and Torchvision
- YOLOv5 and YOLOv8 frameworks
- Google Colab (for training and evaluation)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/yolo-performance-comparison.git
   ```
2. Install the required libraries:
   ```bash
   pip install ultralytics torch torchvision
   ```
3. Train the model:
   - For YOLOv5:
     ```python
     python yolov5_train.py
     ```
   - For YOLOv8:
     ```python
     python yolov8_train.py
     ```
4. Evaluate the model:
   ```python
   python evaluate.py
   ```

## References

- [YOLOv5 Documentation](https://github.com/ultralytics/yolov5)
- [YOLOv8 Documentation](https://github.com/ultralytics/ultralytics)
- Dataset annotations were created using the [LabelImg tool](https://github.com/tzutalin/labelImg).

