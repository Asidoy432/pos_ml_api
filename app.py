"""
POS ML Flask API
Model storage: Hugging Face Hub
Endpoints:
  GET  /                          → health check
  POST /forecast                  → ARIMA 30-day forecast
  POST /recommend                 → Apriori product recommendations
  GET  /stores                    → list available store IDs
  GET  /metrics                   → model performance metrics
"""

import os
import json
import pickle
import logging

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import hf_hub_download

# ── App setup ─────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Hugging Face config ────────────────────────────────────────────────────
HF_REPO_ID  = os.environ.get('HF_REPO_ID',  'YOUR_HF_USERNAME/pos-ml-models')
HF_TOKEN    = os.environ.get('HF_TOKEN',    None)   # only needed for private repos

# Local cache directory on the Render instance
CACHE_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_cache')
ARIMA_CACHE = os.path.join(CACHE_DIR, 'arima_models')
os.makedirs(ARIMA_CACHE, exist_ok=True)


# ── Download helpers ───────────────────────────────────────────────────────
def download_from_hf(filename, local_dir):
    """Download a file from HF Hub if not already cached."""
    local_path = os.path.join(local_dir, os.path.basename(filename))
    if os.path.exists(local_path):
        return local_path
    logger.info(f'Downloading {filename} from HF Hub...')
    downloaded = hf_hub_download(
        repo_id   = HF_REPO_ID,
        filename  = filename,
        token     = HF_TOKEN,
        local_dir = local_dir,
    )
    logger.info(f'  saved to {downloaded}')
    return downloaded


def load_metadata():
    path = download_from_hf('model_metadata.json', CACHE_DIR)
    with open(path) as f:
        return json.load(f)

def load_rules():
    path = download_from_hf('apriori_rules.csv', CACHE_DIR)
    return pd.read_csv(path)

def load_arima(store_id):
    for sid in [store_id, store_id.upper(), store_id.lower()]:
        filename   = f'arima_models/arima_{sid}.pkl'
        local_path = os.path.join(ARIMA_CACHE, f'arima_{sid}.pkl')
        logger.info(f'[load_arima] sid={sid} exists={os.path.exists(local_path)} repo={HF_REPO_ID}')
        if os.path.exists(local_path):
            try:
                with open(local_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f'[load_arima] Unpickle failed {local_path}: {e}')
                return None
        try:
            logger.info(f'[load_arima] Downloading {filename} from HF...')
            download_from_hf(filename, ARIMA_CACHE)
            logger.info(f'[load_arima] Download done, exists={os.path.exists(local_path)}')
            if os.path.exists(local_path):
                with open(local_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f'[load_arima] Download failed {filename}: {e}')
    logger.error(f'[load_arima] No model for store_id={store_id} HF_REPO_ID={HF_REPO_ID}')
    return None


# ── Load shared assets at startup ─────────────────────────────────────────
logger.info('Loading metadata and Apriori rules from Hugging Face...')
try:
    metadata = load_metadata()
    rules_df = load_rules()
    logger.info(f'Loaded {len(rules_df)} association rules')
    logger.info(f'Stores: {metadata.get("arima", {}).get("stores", [])}')
except Exception as e:
    logger.error(f'Startup load error: {e}')
    metadata = {}
    rules_df = pd.DataFrame()

# ── Pre-download ALL ARIMA models at startup ───────────────────────────────
# This avoids per-request download timeouts on Render free tier.
# Models are large (10–30 MB each via LFS/XET) so we load them all at boot.
arima_models_cache = {}

def preload_all_arima():
    stores = metadata.get('arima', {}).get('stores', [])
    logger.info(f'Pre-loading {len(stores)} ARIMA models at startup...')
    for store_id in stores:
        try:
            model = load_arima(store_id)
            if model is not None:
                arima_models_cache[store_id] = model
                logger.info(f'  [OK] {store_id}')
            else:
                logger.warning(f'  [FAIL] {store_id} - model is None')
        except Exception as e:
            logger.error(f'  [ERROR] {store_id}: {e}')
    logger.info(f'Pre-load complete: {len(arima_models_cache)}/{len(stores)} models loaded')

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
            {'store_id': s,
             'mae' : metrics.get(s, {}).get('mae'),
             'rmse': metrics.get(s, {}).get('rmse'),
             'mape': metrics.get(s, {}).get('mape')}
            for s in stores
        ]
    })


@app.route('/forecast', methods=['POST'])
def forecast():
    body          = request.get_json(force=True, silent=True) or {}
    store_id      = body.get('store_id', '').strip()  # keep original casing to match model filenames
    forecast_days = min(int(body.get('forecast_days', 30)), 90)

    if not store_id:
        return jsonify({'error': 'store_id is required'}), 400

    # Check in-memory cache first (preloaded at startup), then try downloading
    model = arima_models_cache.get(store_id) or load_arima(store_id)
    if model is not None:
        arima_models_cache[store_id] = model  # cache for next request
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


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
      


@app.route('/debug', methods=['GET'])
def debug():
    """Quick diagnostics endpoint - remove in production"""
    import glob
    cached_files = glob.glob(os.path.join(ARIMA_CACHE, '*.pkl'))
    return jsonify({
        'hf_repo_id'   : HF_REPO_ID,
        'hf_token_set' : HF_TOKEN is not None,
        'cache_dir'    : ARIMA_CACHE,
        'cached_models': [os.path.basename(f) for f in cached_files],
        'preloaded_in_memory': list(arima_models_cache.keys()),
        'metadata_keys': list(metadata.keys()),
        'stores_in_meta': metadata.get('arima', {}).get('stores', []),
        'rules_loaded' : len(rules_df),
    })
  
