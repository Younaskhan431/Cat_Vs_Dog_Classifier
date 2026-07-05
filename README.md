# 🐾 Cat vs Dog Classifier

A deep learning image classifier that identifies whether an uploaded photo contains a cat or a dog, built with TensorFlow/Keras and deployed as an interactive Streamlit web app.

## Overview

This project compares three modeling approaches on the same dataset and deploys the best-performing one:

| Model                   | Approach                                  |
|--------------------------|--------------------------------------------|
| Custom CNN               | Built from scratch (3 conv blocks)         |
| MobileNetV2 (frozen)     | Transfer learning, frozen base             |
| MobileNetV2 (fine-tuned) | Transfer learning + fine-tuning last 30 layers |

The **fine-tuned MobileNetV2** model performed best and is the one powering the deployed app.

## Demo

Upload any photo of a single cat or dog and the app returns:
- The predicted class (Cat / Dog)
- A confidence score
- A visual confidence breakdown between both classes

## Tech Stack

- **Model training:** TensorFlow / Keras (Google Colab)
- **Architecture:** MobileNetV2 (transfer learning + fine-tuning)
- **Deployment:** Streamlit
- **Other:** NumPy, Pillow

## Project Structure

```
├── model/
│   └── best_fine_tuned_mobilenet.keras   # Trained model weights
├── app.py                                # Streamlit web app
├── Cats_vs_Dogs_classifier.ipynb         # Full training notebook
├── requirements.txt                      # Python dependencies
└── README.md
```

## Running Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/Younaskhan431/cats-vs-dogs-classifier.git
   cd cats-vs-dogs-classifier
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Model Details

- **Input size:** 224 × 224 × 3
- **Preprocessing:** Normalization handled internally via a `Rescaling(1./255)` layer built into the model
- **Output:** Sigmoid activation (binary classification — Cat = 0, Dog = 1)
- **Training:** Data augmentation (flip, rotation, zoom, contrast), fine-tuned on the last 30 layers of MobileNetV2 with a low learning rate

## Author

**Younas Khan**
