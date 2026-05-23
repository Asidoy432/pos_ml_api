!pip install -q flask-cors
import os
import json
import pickle
import logging
import requests
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HF_REPO_ID = os.environ.get('HF_REPO_ID', 'Asidoy432/pos-ml-models')
HF_TOKEN = os.environ.get('HF_TOKEN', None)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd(), 'model_cache')
ARIMA_CACHE = os.path.join(CACHE_DIR, 'arima_models')
os.makedirs(ARIMA_CACHE, exist_ok=True)

def hf_direct_url(filename):
    return f'https://huggingface.co/{HF_REPO_ID}/resolve/main/{filename}'

def download_from_hf(filename, local_dir):
    base_name = os.path.basename(filename)
    local_path = os.path.join(local_dir, base_name)
    
    if os.path.exists(local_path):
        return local_path

    url = hf_direct_url(filename)
    headers = {'Authorization': f'Bearer {HF_TOKEN}'} if HF_TOKEN else {}
    
    logger.info(f'[download] Fetching {url}')
    resp = requests.get(url, headers=headers, stream=True, timeout=120)
    resp.raise_for_status()

    with open(local_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    return local_path

def load_metadata():
    path = download_from_hf('model_metadata.json', CACHE_DIR)
    with open(path) as f:
        return json.load(f)

def load_rules():
    path = download_from_hf('apriori_rules.csv', CACHE_DIR)
    return pd.read_csv(path)

def load_arima(store_id):
    for sid in dict.fromkeys([store_id, store_id.upper(), store_id.lower()]):
        hf_filename = f'arima_models/arima_{sid}.pkl'
        local_path = os.path.join(ARIMA_CACHE, f'arima_{sid}.pkl')

        if not os.path.exists(local_path):
            try:
                download_from_hf(hf_filename, ARIMA_CACHE)
            except Exception as e:
                continue

        if os.path.exists(local_path):
            try:
                with open(local_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f'Unpickle error: {e}')
                os.remove(local_path)
    return None

# --- Startup Load ---
metadata = {}
rules_df = pd.DataFrame()
arima_models_cache = {}

try:
    metadata = load_metadata()
    rules_df = load_rules()
    stores = metadata.get('arima', {}).get('stores', [])
    for sid in stores:
        m = load_arima(sid)
        if m: arima_models_cache[sid] = m
except Exception as e:
    logger.error(f'Startup Error: {e}')

@app.route('/')
def health():
    return jsonify({'status': 'ok', 'models_loaded': len(arima_models_cache)})

@app.route('/forecast', methods=['POST'])
def forecast():
    body = request.get_json(force=True, silent=True) or {}
    store_id = body.get('store_id', '').strip().upper()
    days = min(int(body.get('forecast_days', 30)), 90)
    
    model = arima_models_cache.get(store_id) or load_arima(store_id)
    if not model:
        return jsonify({'error': f'Model {store_id} not found'}), 404

    try:
        vals = model.predict(n_periods=days)
        dates = pd.date_range(pd.Timestamp.today() + pd.Timedelta(days=1), periods=days)
        res = [{'date': str(d.date()), 'predicted_sales': round(max(0, float(v)), 2)} for d, v in zip(dates, vals)]
        return jsonify({'store_id': store_id, 'forecast': res})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/recommend', methods=['POST'])
def recommend():
    body = request.get_json(force=True, silent=True) or {}
    cart = body.get('cart_items', [])
    top_n = int(body.get('top_n', 5))

    if rules_df.empty or not cart:
        return jsonify({'cart_items': cart, 'recommendations': []})

    cart_set = set(cart)
    matches = []
    for _, row in rules_df.iterrows():
        ant = set(row['antecedents'].split(', '))
        if ant.issubset(cart_set):
            matches.append({
                'recommended_products': row['consequents'],
                'confidence': round(float(row['confidence']), 4),
                'lift': round(float(row['lift']), 4)
            })
    
    matches = sorted(matches, key=lambda x: x['lift'], reverse=True)
    return jsonify({'cart_items': cart, 'recommendations': matches[:top_n]})

@app.route('/debug')
def debug():
    return jsonify({
        'cached_files': os.listdir(ARIMA_CACHE),
        'in_memory': list(arima_models_cache.keys()),
        'metadata': metadata
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
