# Flower Classification with CNN

## Project Description
This project implements a Convolutional Neural Network (CNN) to classify flower images into five categories:
daisy, dandelion, rose, sunflower, and tulip.

## Dataset
The dataset was obtained from Kaggle:
Flowers Dataset (imsparsh)

Total images: ~4300  
Classes: 5

## Technologies
- Python 3.10
- TensorFlow / Keras
- CNN (Convolutional Neural Networks)
- Conda Environment

## Model Architecture
- Conv2D + ReLU
- MaxPooling
- Fully Connected (Dense)
- Softmax Output Layer

## Training Results
- Training Accuracy: ~99%
- Validation Accuracy: ~63%

The difference between training and validation accuracy indicates overfitting.

## How to Run
```bash
conda create -n proje python=3.10
conda activate proje
pip install -r requirements.txt
python main.py
