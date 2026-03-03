# Facial Expression Recognition (FER) using CNN

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-5C3EE8?style=flat&logo=opencv&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?style=flat&logo=Kaggle&logoColor=white)

This repository contains a Deep Learning pipeline designed to recognize human emotions from facial images in real-time. By combining a custom Convolutional Neural Network (CNN) with OpenCV's computer vision capabilities, the system can classify emotions into seven distinct categories with high precision.

---

## 📂 Dataset Information

The model is trained on the **FER-2013** (Facial Expression Recognition) dataset available on Kaggle.

* **Dataset Link**: [Face Expression Recognition Dataset (Kaggle)](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset)
* **Total Images**: ~35,887 grayscale images.
* **Resolution**: $48 \times 48$ pixels (Resized to $96 \times 96$ for this project).
* **Classes**: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise.

---

## 🛠️ Technology Stack & Logic

### 1. Haar Cascade Classifier (`haarcascade_frontalface_default.xml`)
In the real-time detection script, this XML file is used to perform **Face Detection**. 
* **How it works**: It uses the Viola-Jones algorithm to scan the image for "Haar-like features" (e.g., the bridge of the nose is lighter than the eyes). 
* **Purpose**: It allows the system to find the location $(x, y, w, h)$ of a face in a cluttered background, crop that specific region, and pass it to our CNN for emotion classification.

### 2. Convolutional Neural Network (CNN)
The heart of the project is a Sequential CNN architecture:
* **Feature Extraction**: Three layers of `Conv2D` to identify spatial patterns.
* **Optimization**: `Adam` optimizer with a `ReduceLROnPlateau` callback to lower the learning rate when validation loss stops improving.
* **Regularization**: `Dropout` (20%) to prevent the model from memorizing the training data (overfitting).

---

## 📊 Performance Scores

After training and fine-tuning, the model achieved an overall **Accuracy of 90%**.

### Classification Report
| Emotion | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Surprise** | 0.93 | 0.92 | 0.92 | 3205 |
| **Sad** | 0.87 | 0.88 | 0.87 | 4938 |
| **Happy** | 0.94 | 0.94 | 0.94 | 7164 |
| **Fear** | 0.87 | 0.86 | 0.87 | 4103 |
| **Disgust** | 0.93 | 0.86 | 0.90 | 436 |
| **Angry** | 0.88 | 0.87 | 0.88 | 3993 |
| **Neutral** | 0.88 | 0.89 | 0.89 | 4982 |

### Confusion Matrix Insights
The model shows exceptional performance in identifying **Happy** and **Surprise** emotions, which have very distinct facial features. Minor confusion exists between **Sad** and **Neutral**, which is a common challenge in the FER-2013 dataset.

---

## 💡 Key Takeaways
* The use of **Learning Rate Reduction** improved final accuracy significantly.
* **Float16** precision was used in the `x_train` array to save memory while handling a large number of images.
