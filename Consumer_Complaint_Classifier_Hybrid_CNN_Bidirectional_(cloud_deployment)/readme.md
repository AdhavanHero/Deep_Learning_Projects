# 🗂️ Consumer Complaint Classifier

> **Deep Learning-Based NLP Classification System for Financial Complaints**  
> Capstone Project · Adhavan Ponram · 2026

A production-deployed web application that automatically classifies consumer financial complaints into product categories using a hybrid **CNN-BiLSTM** deep learning model. Built with Flask, trained on CFPB data, and live on AWS EC2.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running the App](#running-the-app)
- [Deployment](#deployment)
- [API & Database](#api--database)
- [Acknowledgements](#acknowledgements)

---

## Overview

Banks and financial institutions receive hundreds of thousands of consumer complaints every year. Manually routing each complaint to the correct product team is slow, expensive, and inconsistent.

This project trains four deep learning NLP architectures and selects the best one to automatically classify complaints into **9 financial product categories** — reducing manual effort and improving response time.

**Key highlights:**
- ~45,000 balanced complaint samples across 9 categories
- CNN-BiLSTM hybrid achieves **~89% classification accuracy**
- Full Flask web app with Xano database integration
- Deployed live on **AWS EC2** with Gunicorn + tmux

---

## Demo

The app accepts a customer complaint in natural language, predicts the financial product category, and displays confidence scores for the top 3 predictions. Each submission is assigned a unique **6-digit Complaint ID** and logged to a cloud database automatically.

**Low-confidence guard:** If the model confidence is below 20%, the app politely asks the user to provide more details rather than returning a potentially incorrect classification.

---

## Dataset

**Source:** [Consumer Financial Protection Bureau (CFPB) — Consumer Complaint Database](https://catalog.data.gov/dataset/consumer-complaint-database)

| Property | Detail |
|---|---|
| Raw file | `complaints.csv` (millions of rows) |
| Columns used | `Product` + `Consumer complaint narrative` |
| Final dataset | ~45,000 rows (balanced) |
| Classes | 9 product categories |
| Samples per class | 5,000 (downsampled) |

### 9 Product Categories

| # | Category |
|---|---|
| 1 | Credit Reporting |
| 2 | Credit Card |
| 3 | Personal / Payday Loan |
| 4 | Mortgage |
| 5 | Debt Collection |
| 6 | Checking / Savings Account |
| 7 | Student Loan |
| 8 | Money Transfer / Services |
| 9 | Vehicle Loan |

### Data Preparation Steps

1. **Chunked loading** — read `complaints.csv` in 100K-row chunks to avoid memory overflow
2. **Drop nulls** — removed rows with no complaint narrative
3. **Category mapping** — unified 25+ verbose product names into 9 clean categories using a custom dictionary
4. **Class balancing** — downsampled each category to 5,000 samples (`random_state=42`)
5. **Text cleaning** — lowercased, removed punctuation, stopwords (NLTK), and special characters
6. **Tokenization & padding** — Keras Tokenizer (top 20K words) + `pad_sequences` (MAX_LEN = 150)

### Known Dataset Challenges

- **Missing narratives** — many rows had no complaint text; dropped entirely
- **Inconsistent labels** — same product appeared with multiple name variants; resolved with a mapping dictionary
- **Class imbalance** — Credit Reporting heavily overrepresented; fixed by downsampling
- **Redacted credentials** — `XXXX` placeholders throughout narratives; treated as non-signal noise

---

## Model Architecture

Four deep learning architectures were trained and compared:

| Trial | Architecture | Notes |
|---|---|---|
| 1 | SimpleRNN | Baseline; struggles with long sequences (vanishing gradient) |
| 2 | LSTM | Better long-range memory than SimpleRNN |
| 3 | BiLSTM | Reads sequences in both directions; improved context |
| 4 | **CNN-BiLSTM (Hybrid)** ✅ | **Best performer — ~81% accuracy** |

### Final Model: CNN-BiLSTM Hybrid

```
Embedding(vocab_size, 150)
    → SpatialDropout1D
    → Conv1D(128, kernel_size=5, relu)   ← local n-gram pattern extraction
    → BiLSTM(128)                         ← global sequence modeling (both directions)
    → Dense(64, relu)
    → Dense(9, softmax)
```

**Key architectural decisions:**
- **SpatialDropout1D** — drops entire feature maps after embedding to prevent over-reliance on specific word positions
- **Conv1D** — extracts local n-gram patterns (e.g., *"account was closed"*, *"charged twice"*) before the recurrent layer
- **Bidirectional LSTM** — doubles context capacity by processing sequences left-to-right and right-to-left simultaneously
- **Intermediate Dense(64)** — translates high-dimensional RNN output to class probabilities more effectively
- **ReduceLROnPlateau** — automatically reduces learning rate when validation loss stalls

---

## Results

| Model | Validation Accuracy |
|---|---|
| SimpleRNN | ~45% |
| LSTM | ~77% |
| BiLSTM | ~79% |
| **CNN-BiLSTM** | **~81%** |

All 4 models are saved as `.h5` files. The tokenizer and label encoder are serialized as `.pkl` files for deployment.

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML / DL | TensorFlow · Keras · NLTK |
| Backend | Python 3 · Flask |
| Frontend | HTML5 · CSS3 (no JS framework) |
| Database | Xano (no-code REST API + cloud DB) |
| Deployment | AWS EC2 (Ubuntu, T3) · Gunicorn · tmux |
| File Transfer | WinSCP (SFTP) · PuTTY (SSH) |

---

## Project Structure

```
consumer-complaint-classifier/
│
├── app.py                              # Flask application
├── style.css                           # Frontend styles
├── templates/
│   └── index.html                      # Main UI template
├── static/
│   └── images/                         # Category avatar images
│       ├── credit_reporting.jpg
│       ├── credit_card.jpg
│       ├── mortgage.jpg
│       └── ...
│
├── model_4_hybrid_cnn_lstm(new).h5     # Trained CNN-BiLSTM model
├── tokenizer.pkl                       # Keras tokenizer
├── label_encoder.pkl                   # Sklearn label encoder
│
├── Python_1.ipynb                      # Training & experimentation notebook
└── README.md
```

> **Note:** The `.h5`, `.pkl` model files are not included in the repository due to size. See setup instructions below.

---

## Setup & Installation

### Prerequisites

- Python 3.8+
- pip

### 1. Clone the repository

```bash
git clone https://github.com/your-username/consumer-complaint-classifier.git
cd consumer-complaint-classifier
```

### 2. Install dependencies

```bash
pip install flask tensorflow keras nltk numpy requests scikit-learn
```

### 3. Download NLTK stopwords

```python
import nltk
nltk.download('stopwords')
```

### 4. Add model artifacts

Place the following files in the project root directory:

```
model_4_hybrid_cnn_lstm(new).h5
tokenizer.pkl
label_encoder.pkl
```

> These files are generated by running the training notebook (`Python_1.ipynb`). Download the CFPB dataset, run the notebook end-to-end, and the files will be saved automatically.

---

## Running the App

```bash
python app.py
```

The app starts on `http://localhost:5000` by default.

You can verify the model loaded correctly by hitting the health endpoint:

```
GET http://localhost:5000/health
```

Expected response:
```json
{ "status": "ok", "model_ready": true }
```

---

## Deployment

The app is deployed on **AWS EC2 (Ubuntu, T3 instance)** using Gunicorn as the production WSGI server.

### Production deployment steps

```bash
# Install Gunicorn
pip install gunicorn

# Start app in a persistent tmux session
tmux new -s complaint-app
gunicorn -w 2 -b 0.0.0.0:5000 app:app

# Detach from tmux (app keeps running after SSH closes)
# Ctrl+B then D
```

File transfer to EC2 was done via **WinSCP** (SFTP), and SSH access via **PuTTY** using the `.pem` key pair.

---

## API & Database

Every complaint submission is automatically logged to a **Xano** cloud database via a REST API call.

**Payload sent to Xano on each prediction:**

```json
{
  "complaint_id": 482931,
  "customer_name": "John Doe",
  "account_number": "12345",
  "complaint_type": "Credit Reporting",
  "complaint_description": "Full complaint text here..."
}
```

Fallback handling: if the Xano API is unreachable (5s timeout), the app catches the error gracefully and still returns the prediction to the user.

> **Note:** The Xano endpoint URL in `app.py` has been masked for public release.

---

## Acknowledgements

- **Dataset:** [Consumer Financial Protection Bureau (CFPB)](https://catalog.data.gov/dataset/consumer-complaint-database) via data.gov
- **Frameworks:** TensorFlow / Keras, Flask, NLTK
- **Infrastructure:** AWS EC2, Xano, Gunicorn

---

*Built as a Capstone Project · 2026 · Adhavan Ponram*
