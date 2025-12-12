# Handwritten Digit Recognition

A simple handwritten digit recognition project built with **TensorFlow**
that trains a model to classify digits (0--9) using image data. This
project uses CSV image data (e.g., MNIST‑style flattened pixel values)
and includes training and evaluation scripts.

## 📌 Project Overview

This repository contains:

-   `trainer.py` --- script to train the digit recognition model\
-   `tester.py` --- script to evaluate or run predictions on new images\
-   `train.csv` / `test.csv` --- labeled and unlabeled digit datasets\
-   `my_model.keras` --- saved trained model\
-   Example images (`Test.png`, `img.png`)

The goal is to build a model that learns to recognize handwritten digits
and can make predictions on new input images.

## 📦 Requirements

Install the necessary dependencies:

``` bash
pip install tensorflow numpy pandas matplotlib
```

## 🧠 Usage

### 🏋️ Train the Model

``` bash
python trainer.py
```

### 🔍 Evaluate / Predict

``` bash
python tester.py --model my_model.keras
```

## 📁 Project Structure

    HandwrittenDigitRecognition/
    ├── .idea/
    ├── Train.csv
    ├── Test.csv
    ├── trainer.py
    ├── tester.py
    ├── my_model.keras
    ├── img.png
    ├── Test.png
    └── README.md

## 📈 Results

Typical digit recognition models trained on MNIST reach high accuracy
(often above 95--99% depending on model complexity and preprocessing).

## 📝 Tips for Improvement

-   Add data preprocessing (normalization, reshaping)
-   Visualize training history
-   Improve model with convolutional layers
-   Add CLI prediction interface

## 📜 License

This project is open source --- feel free to modify and build on it.
