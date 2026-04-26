from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
import os
import pickle
import re
import string
import numpy as np
import requests
import random
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_TRT_LOGGER'] = 'ERROR'

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

app = Flask(__name__, template_folder='templates', static_folder='static')

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model_4_hybrid_cnn_lstm(new).h5')
TOKENIZER_PATH = os.path.join(BASE_DIR, 'tokenizer.pkl')
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, 'label_encoder.pkl')
MAX_SEQUENCE_LENGTH = 150

# Download stopwords
try:
    stopwords.words('english')
except:
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Load model and preprocessors
model = None
tokenizer = None
label_encoder = None
model_loaded = False

def load_resources():
    global model, tokenizer, label_encoder, model_loaded
    try:
        if os.path.exists(MODEL_PATH):
            model = load_model(MODEL_PATH)
        if os.path.exists(TOKENIZER_PATH):
            with open(TOKENIZER_PATH, 'rb') as f:
                tokenizer = pickle.load(f)
        if os.path.exists(LABEL_ENCODER_PATH):
            with open(LABEL_ENCODER_PATH, 'rb') as f:
                label_encoder = pickle.load(f)
        model_loaded = (model is not None and tokenizer is not None and label_encoder is not None)
    except Exception as e:
        print(f"Error loading resources: {e}")
        model_loaded = False

# Load resources when module is imported (for both Flask running and testing)
load_resources()

# Absolute path to this module's static images directory
STATIC_IMAGES_DIR = os.path.join(BASE_DIR, 'static', 'images')
# Clean text function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", ' ', text)
    text = re.sub(r"\d+", ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = [w for w in text.split() if w not in stop_words and len(w) > 2]
    return ' '.join(words)

@app.route('/', methods=['GET', 'POST'])
def index():
    # Server-side form handling so the app works with only Flask + HTML + CSS (no JS)
    if request.method == 'POST':
        if not model_loaded:
            return render_template('index.html', model_ready=model_loaded, error='Model is not loaded. Please ensure model artifacts are present.')

        user_text = request.form.get('text', '').strip()
        if not user_text:
            return render_template('index.html', model_ready=model_loaded, error='Please enter a complaint text.')

        try:
            cleaned_text = clean_text(user_text)
            seq = tokenizer.texts_to_sequences([cleaned_text])
            padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

            probs = model.predict(padded, verbose=0)
            predicted_idx = int(np.argmax(probs, axis=1)[0])
            confidence = float(np.max(probs))
            
            # Check for low confidence (< 20%)
            if confidence < 0.20:
                polite_msg = "We are so sorry for the inconvenience, and we are here to help you! However, we couldn't clearly classify your complaint. Could you please provide a few more details about your queries so we can assist you better?"
                return render_template('index.html', model_ready=model_loaded, error=polite_msg)

            predicted_label = label_encoder.inverse_transform([predicted_idx])[0]

            # Xano Integration
            customer_name = request.form.get('customer_name', 'Unknown')
            account_number = request.form.get('account_number', '00000')
            complaint_id = random.randint(100000, 999999)
            
            xano_data = {
                "complaint_id": complaint_id,
                "customer_name": customer_name,
                "account_number": account_number,
                "complaint_type": predicted_label,
                "complaint_description": user_text
            }
            try:
                resp = requests.post(
                    "https://your-xano-instance.xano.io/api:endpoint/register_complaint", # MASKED FOR GITHUB
                    json=xano_data, timeout=5
                )
                print(f"Xano status: {resp.status_code}, response: {resp.text}")
            except Exception as e:
                print(f"Xano error: {e}")

            # Top 3 predictions
            top_3_indices = np.argsort(probs[0])[-3:][::-1]
            top_3 = [
                {
                    'label': label_encoder.inverse_transform([int(idx)])[0],
                    'confidence': float(probs[0][int(idx)])
                }
                for idx in top_3_indices
            ]

            # Simple avatar mapping (anime-style placeholders in static/images/)
            avatar_map = {
                # credit reporting
                'credit_reporting': 'credit_reporting.jpg',
                'creditreport': 'credit_reporting.jpg',
                'credit_report': 'credit_reporting.jpg',

                # student loan
                'student_loan': 'student_loan.jpg',
                'studentloan': 'student_loan.jpg',

                # credit card
                'credit_card': 'credit_card.jpg',
                'creditcard': 'credit_card.jpg',

                # debt collection
                'debt_collection': 'debt_collection.jpg',
                'debtcollection': 'debt_collection.jpg',

                # money transfer / wire
                'money_transfer': 'money_transfer.jpg',
                'moneytransfer': 'money_transfer.jpg',
                'wire_transfer': 'money_transfer.jpg',

                # mortgage
                'mortgage': 'mortgage.jpg',

                # personal / payday loan
                'personal_loan': 'personal_loan.jpg',
                'personalpaydayloan': 'personal_loan.jpg',
                'personal_payday_loan': 'personal_loan.jpg',
                'personal_payday': 'personal_loan.jpg',
                'personal': 'personal_loan.jpg',

                # vehicle / auto loan
                'vehicle_loan': 'vehicle_loan.jpg',
                'vehicleloan': 'vehicle_loan.jpg',
                'auto_loan': 'vehicle_loan.jpg',

                # bank account
                'bank_account': 'bank_account.jpg',
                'bankaccount': 'bank_account.jpg',
                'bank': 'bank_account.jpg'
            }

            # Attach avatar filename to each top-3 item (normalize label)
            for item in top_3:
                # normalize label to simple key: lower, non-alnum -> underscore
                key = re.sub(r'[^a-z0-9]+', '_', item['label'].lower()).strip('_')
                # Preferred extensions: jpg, png, svg
                candidate = avatar_map.get(key)
                found = None
                if candidate:
                    # try candidate as-is first (resolve to absolute static images dir)
                    paths = [os.path.join(STATIC_IMAGES_DIR, candidate)]
                else:
                    paths = [
                        os.path.join(STATIC_IMAGES_DIR, f"{key}.jpg"),
                        os.path.join(STATIC_IMAGES_DIR, f"{key}.png"),
                        os.path.join(STATIC_IMAGES_DIR, f"{key}.svg")
                    ]

                # if candidate provided, also try jpg/png variants
                if candidate:
                    base = os.path.splitext(candidate)[0]
                    paths = [
                        os.path.join(STATIC_IMAGES_DIR, f"{base}.jpg"),
                        os.path.join(STATIC_IMAGES_DIR, f"{base}.png"),
                        os.path.join(STATIC_IMAGES_DIR, f"{base}.svg"),
                        os.path.join(STATIC_IMAGES_DIR, candidate)
                    ]

                for p in paths:
                    if os.path.exists(p):
                        found = os.path.basename(p)
                        break

                # If not found yet, try fuzzy-matching files in the pics folder
                if not found:
                    pics_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pics'))
                    if os.path.exists(pics_dir):
                        for fname in os.listdir(pics_dir):
                            norm = re.sub(r'[^a-z0-9]+', '_', fname.lower()).strip('_')
                            if norm == key or key in norm or norm in key:
                                found = fname
                                break

                if found:
                    # Build avatar_url similarly to earlier logic
                    static_path = os.path.join('static', 'images', found)
                    if os.path.exists(static_path):
                        item['avatar_url'] = url_for('static', filename=f'images/{found}')
                    else:
                        pics_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pics'))
                        if os.path.exists(os.path.join(pics_dir, found)):
                            item['avatar_url'] = url_for('pics_file', filename=found)
                        else:
                            item['avatar_url'] = url_for('static', filename='images/default.svg')
                else:
                    item['avatar_url'] = url_for('static', filename='images/default.svg')

                # Extra safety: if label mentions 'personal' or 'payday', force personal_loan image
                low_label = item['label'].lower()
                if 'personal' in low_label or 'payday' in low_label:
                    personal_path = os.path.join('static', 'images', 'personal_loan.jpg')
                    if os.path.exists(personal_path):
                        item['avatar_url'] = url_for('static', filename='images/personal_loan.jpg')

                pass

            confidence_percent = f"{confidence * 100:.1f}%"

            return render_template(
                'index.html',
                model_ready=model_loaded,
                prediction=predicted_label,
                confidence_percent=confidence_percent,
                top3=top_3,
                complaint_id=complaint_id
            )
        except Exception as e:
            return render_template('index.html', model_ready=model_loaded, error=f'Prediction error: {str(e)}')

    return render_template('index.html', model_ready=model_loaded)


@app.route('/pics/<path:filename>')
def pics_file(filename):
    # Serve images from the sibling `pics` folder placed next to `bilstm_app`
    pics_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pics'))
    return send_from_directory(pics_dir, filename)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_ready': model_loaded})

if __name__ == '__main__':
    if model_loaded:
        print("Model and preprocessors loaded successfully!")
    else:
        print("Warning: Model or preprocessors not found. Please ensure these files exist:")
        print(f"  - {MODEL_PATH}")
        print(f"  - {TOKENIZER_PATH}")
        print(f"  - {LABEL_ENCODER_PATH}")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
