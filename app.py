"""
POS ML Flask API
Model storage: Hugging Face Hub
Endpoints:
  GET  /                          → health check
  POST /forecast                  → ARIMA 30-day forecast
  POST /recommend                 → Apriori product recommendations
  GET  /stores                    → list available store IDs
  GET  /metrics                   → model performance metrics
  GET  /debug                     → diagnostics
"""

import os
import json
import pickle
import logging
import requests

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── App setup ─────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Hugging Face config ────────────────────────────────────────────────────
HF_REPO_ID = os.environ.get('HF_REPO_ID', 'Asidoy432/pos-ml-models')
HF_TOKEN   = os.environ.get('HF_TOKEN', None)

# Local cache directory on the Render instance
CACHE_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_cache')
ARIMA_CACHE = os.path.join(CACHE_DIR, 'arima_models')
os.makedirs(ARIMA_CACHE, exist_ok=True)


# ── Download helpers ───────────────────────────────────────────────────────
def hf_direct_url(filename):
    """Build direct HTTPS download URL for HuggingFace (bypasses XET/LFS issues)."""
    return f'https://huggingface.co/{HF_REPO_ID}/resolve/main/{filename}'

def download_from_hf(filename, local_dir):
    """Download a file from HF Hub via direct HTTPS — works with XET/LFS storage."""
    local_path = os.path.join(local_dir, os.path.basename(filename))
    if os.path.exists(local_path):
        logger.info(f'[download] Already cached: {local_path}')
        return local_path

    url = hf_direct_url(filename)
    headers = {}
    if HF_TOKEN:
        headers['Authorization'] = f'Bearer {HF_TOKEN}'

    logger.info(f'[download] GET {url}')
    resp = requests.get(url, headers=headers, stream=True, timeout=120)
    resp.raise_for_status()

    size = 0
    with open(local_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            size += len(chunk)

    logger.info(f'[download] Saved {local_path} ({size/1024/1024:.1f} MB)')
    return local_path


def load_metadata():
    path = download_from_hf('model_metadata.json', CACHE_DIR)
    with open(path) as f:
        return json.load(f)

def load_rules():
    path = download_from_hf('apriori_rules.csv', CACHE_DIR)
    return pd.read_csv(path)

def load_arima(store_id):
    """Download and unpickle an ARIMA model. Tries original casing, upper, lower."""
    for sid in dict.fromkeys([store_id, store_id.upper(), store_id.lower()]):
        filename   = f'arima_models/arima_{sid}.pkl'
        local_path = os.path.join(ARIMA_CACHE, f'arima_{sid}.pkl')
        logger.info(f'[load_arima] Trying {sid} | cached={os.path.exists(local_path)}')

        if not os.path.exists(local_path):
            try:
                download_from_hf(filename, ARIMA_CACHE)
            except Exception as e:
                logger.warning(f'[load_arima] Download failed for {filename}: {e}')
                continue

        if os.path.exists(local_path):
            try:
                with open(local_path, 'rb') as f:
                    model = pickle.load(f)
                logger.info(f'[load_arima] Loaded {sid} OK')
                return model
            except Exception as e:
                logger.error(f'[load_arima] Unpickle failed {local_path}: {e}')
                os.remove(local_path)  # remove corrupt file

    logger.error(f'[load_arima] No model for store_id={store_id}')
    return None


# ── Load shared assets at startup ─────────────────────────────────────────
logger.info('Loading metadata and Apriori rules...')
try:
    metadata = load_metadata()
    rules_df = load_rules()
    logger.info(f'Loaded {len(rules_df)} association rules')
    logger.info(f'Stores: {metadata.get("arima", {}).get("stores", [])}')
except Exception as e:
    logger.error(f'Startup load error: {e}')
    metadata = {}
    rules_df = pd.DataFrame()

# ── Pre-load all ARIMA models at startup ──────────────────────────────────
arima_models_cache = {}

def preload_all_arima():
    stores = metadata.get('arima', {}).get('stores', [])
    logger.info(f'Pre-loading {len(stores)} ARIMA models...')
    for store_id in stores:
        try:
            model = load_arima(store_id)
            if model is not None:
                arima_models_cache[store_id] = model
                logger.info(f'  [OK] {store_id}')
            else:
                logger.warning(f'  [FAIL] {store_id}')
        except Exception as e:
            logger.error(f'  [ERROR] {store_id}: {e}')
    logger.info(f'Pre-load done: {len(arima_models_cache)}/{len(stores)} loaded')

preload_all_arima()


# ── Helpers ────────────────────────────────────────────────────────────────
def recommend_products(cart_items, top_n=5):
    if rules_df.empty:
        return []
    cart_set = set(cart_items)
    matches  = []
    for _, row in rules_df.iterrows():
        ant = set(row['antecedents'].split(', '))
        con = set(row['consequents'].split(', '))
        if ant.issubset(cart_set) and not con.issubset(cart_set):
            matches.append({
                'recommended_products': row['consequents'],
                'confidence'          : round(float(row['confidence']), 4),
                'lift'                : round(float(row['lift']), 4),
                'support'             : round(float(row['support']), 4),
            })
    return sorted(matches, key=lambda x: x['lift'], reverse=True)[:top_n]


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET'])
def health_check():
    stores = metadata.get('arima', {}).get('stores', [])
    return jsonify({
        'status' : 'ok',
        'message': 'POS ML API is running',
        'models' : {
            'arima'  : f'{len(stores)} store models available',
            'apriori': f'{len(rules_df)} rules loaded',
        },
        'hf_repo': HF_REPO_ID,
        'version': metadata.get('version', '1.0.0'),
    })


@app.route('/stores', methods=['GET'])
def get_stores():
    stores  = metadata.get('arima', {}).get('stores', [])
    metrics = metadata.get('arima', {}).get('metrics', {})
    return jsonify({
        'stores': [
            {
                'store_id': s,
                'mae' : metrics.get(s, {}).get('mae'),
                'rmse': metrics.get(s, {}).get('rmse'),
                'mape': metrics.get(s, {}).get('mape'),
            }
            for s in stores
        ]
    })


@app.route('/forecast', methods=['POST'])
def forecast():
    body          = request.get_json(force=True, silent=True) or {}
    store_id      = body.get('store_id', '').strip()
    forecast_days = min(int(body.get('forecast_days', 30)), 90)

    if not store_id:
        return jsonify({'error': 'store_id is required'}), 400

    # Use in-memory cache first, fallback to download
    model = arima_models_cache.get(store_id) or load_arima(store_id)
    if model is not None:
        arima_models_cache[store_id] = model
    if model is None:
        available = list(arima_models_cache.keys()) or metadata.get('arima', {}).get('stores', [])
        return jsonify({'error': f'No model for "{store_id}"', 'available': available}), 404

    try:
        forecast_vals, conf_int = model.predict(n_periods=forecast_days, return_conf_int=True)
        last_date    = pd.Timestamp.today().normalize()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_days)

        result = []
        for date, val, lo, hi in zip(future_dates, forecast_vals, conf_int[:, 0], conf_int[:, 1]):
            result.append({
                'date'            : str(date.date()),
                'predicted_sales' : round(max(0.0, float(val)), 2),
                'lower_bound'     : round(max(0.0, float(lo)),  2),
                'upper_bound'     : round(max(0.0, float(hi)),  2),
            })
        return jsonify({'store_id': store_id, 'forecast': result})
    except Exception as e:
        logger.error(f'Forecast error: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/recommend', methods=['POST'])
def recommend():
    body       = request.get_json(force=True, silent=True) or {}
    cart_items = body.get('cart_items', [])
    top_n      = int(body.get('top_n', 5))

    if not cart_items or not isinstance(cart_items, list):
        return jsonify({'error': 'cart_items must be a non-empty list'}), 400
    if rules_df.empty:
        return jsonify({'error': 'Apriori rules not loaded'}), 503

    try:
        recs = recommend_products(cart_items, top_n=top_n)
        return jsonify({'cart_items': cart_items, 'recommendations': recs})
    except Exception as e:
        logger.error(f'Recommend error: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/metrics', methods=['GET'])
def get_metrics():
    return jsonify({
        'arima_metrics'  : metadata.get('arima', {}).get('metrics', {}),
        'apriori_summary': {
            'total_rules'   : metadata.get('apriori', {}).get('total_rules'),
            'min_support'   : metadata.get('apriori', {}).get('min_support'),
            'min_confidence': metadata.get('apriori', {}).get('min_confidence'),
        },
        'trained_on': metadata.get('trained_on'),
    })


@app.route('/debug', methods=['GET'])
def debug():
    import glob
    cached_files = glob.glob(os.path.join(ARIMA_CACHE, '*.pkl'))
    return jsonify({
        'hf_repo_id'         : HF_REPO_ID,
        'hf_token_set'       : HF_TOKEN is not None,
        'cache_dir'          : ARIMA_CACHE,
        'cached_models'      : sorted([os.path.basename(f) for f in cached_files]),
        'preloaded_in_memory': sorted(list(arima_models_cache.keys())),
        'metadata_keys'      : list(metadata.keys()),
        'stores_in_meta'     : metadata.get('arima', {}).get('stores', []),
        'rules_loaded'       : len(rules_df),
    })


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)


@app.route('/test-download', methods=['GET'])
def test_download():
    """Test downloading one model file - for debugging only"""
    import traceback
    store_id = request.args.get('store', 'BAR-01')
    filename = f'arima_models/arima_{store_id}.pkl'
    url = hf_direct_url(filename)
    headers = {}
    if HF_TOKEN:
        headers['Authorization'] = f'Bearer {HF_TOKEN}'
    try:
        resp = requests.get(url, headers=headers, stream=True, timeout=30)
        first_bytes = b''
        for chunk in resp.iter_content(chunk_size=1024):
            first_bytes = chunk
            break
        return jsonify({
            'url'         : url,
            'status_code' : resp.status_code,
            'headers'     : dict(resp.headers),
            'first_bytes' : first_bytes[:20].hex(),
            'token_used'  : HF_TOKEN is not None,
        })
    except Exception as e:
        return jsonify({
            'url'  : url,
            'error': str(e),
            'trace': traceback.format_exc(),
        }), 500
