"""
POS ML Flask API
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

# ── App setup ─────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)   # allow cross-origin requests from your PHP frontend

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
ARIMA_DIR     = os.path.join(BASE_DIR, 'arima_models')
RULES_CSV     = os.path.join(BASE_DIR, 'apriori_rules.csv')
ENCODER_PKL   = os.path.join(BASE_DIR, 'apriori_encoder.pkl')
METADATA_JSON = os.path.join(BASE_DIR, 'model_metadata.json')


# ── Load assets at startup ─────────────────────────────────────────────────
def load_metadata():
    if os.path.exists(METADATA_JSON):
        with open(METADATA_JSON) as f:
            return json.load(f)
    return {}

def load_rules():
    if os.path.exists(RULES_CSV):
        return pd.read_csv(RULES_CSV)
    logger.warning('apriori_rules.csv not found')
    return pd.DataFrame()

def load_arima(store_id):
    path = os.path.join(ARIMA_DIR, f'arima_{store_id}.pkl')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

metadata   = load_metadata()
rules_df   = load_rules()
logger.info(f'Loaded {len(rules_df)} association rules')
logger.info(f'Available ARIMA stores: {list(metadata.get("arima", {}).get("stores", []))}')


# ── Helpers ────────────────────────────────────────────────────────────────
def recommend_products(cart_items, top_n=5):
    """Return product recommendations based on cart contents."""
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
    matches = sorted(matches, key=lambda x: x['lift'], reverse=True)
    return matches[:top_n]


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET'])
def health_check():
    """Health check — confirms API is running."""
    stores = metadata.get('arima', {}).get('stores', [])
    return jsonify({
        'status' : 'ok',
        'message': 'POS ML API is running',
        'models' : {
            'arima'  : f'{len(stores)} store models loaded',
            'apriori': f'{len(rules_df)} rules loaded',
        },
        'version': metadata.get('version', '1.0.0'),
    })


@app.route('/stores', methods=['GET'])
def get_stores():
    """Return list of stores that have trained ARIMA models."""
    stores  = metadata.get('arima', {}).get('stores', [])
    metrics = metadata.get('arima', {}).get('metrics', {})
    return jsonify({
        'stores': [
            {
                'store_id': s,
                'mae'     : metrics.get(s, {}).get('mae'),
                'rmse'    : metrics.get(s, {}).get('rmse'),
                'mape'    : metrics.get(s, {}).get('mape'),
            }
            for s in stores
        ]
    })


@app.route('/forecast', methods=['POST'])
def forecast():
    """
    Predict daily sales for a given store.

    Request JSON:
    {
        "store_id"     : "BAR-01",
        "forecast_days": 30          // optional, default 30, max 90
    }

    Response JSON:
    {
        "store_id"   : "BAR-01",
        "forecast"   : [
            {"date": "2025-01-01", "predicted_sales": 1234.56,
             "lower_bound": 900.00, "upper_bound": 1500.00},
            ...
        ]
    }
    """
    body = request.get_json(force=True, silent=True) or {}

    store_id      = body.get('store_id', '').strip().upper()
    forecast_days = min(int(body.get('forecast_days', 30)), 90)

    if not store_id:
        return jsonify({'error': 'store_id is required'}), 400

    model = load_arima(store_id)
    if model is None:
        available = metadata.get('arima', {}).get('stores', [])
        return jsonify({
            'error'    : f'No ARIMA model found for store_id "{store_id}"',
            'available': available
        }), 404

    try:
        forecast_vals, conf_int = model.predict(
            n_periods=forecast_days,
            return_conf_int=True
        )
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
        logger.error(f'Forecast error for {store_id}: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Recommend products to add to a cart using Apriori association rules.

    Request JSON:
    {
        "cart_items": ["Ganador Sardines in Tomato Sauce 155g", "Instant Coffee 3-in-1"],
        "top_n"     : 5    // optional, default 5
    }

    Response JSON:
    {
        "cart_items"     : [...],
        "recommendations": [
            {"recommended_products": "...", "confidence": 0.45, "lift": 2.1, "support": 0.03},
            ...
        ]
    }
    """
    body = request.get_json(force=True, silent=True) or {}

    cart_items = body.get('cart_items', [])
    top_n      = int(body.get('top_n', 5))

    if not cart_items or not isinstance(cart_items, list):
        return jsonify({'error': 'cart_items must be a non-empty list'}), 400

    if rules_df.empty:
        return jsonify({'error': 'Apriori rules not loaded'}), 503

    try:
        recs = recommend_products(cart_items, top_n=top_n)
        return jsonify({
            'cart_items'     : cart_items,
            'recommendations': recs,
        })
    except Exception as e:
        logger.error(f'Recommend error: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Return training metrics for all models."""
    arima_metrics   = metadata.get('arima', {}).get('metrics', {})
    apriori_meta    = metadata.get('apriori', {})
    return jsonify({
        'arima_metrics'  : arima_metrics,
        'apriori_summary': {
            'total_rules'   : apriori_meta.get('total_rules'),
            'min_support'   : apriori_meta.get('min_support'),
            'min_confidence': apriori_meta.get('min_confidence'),
        },
        'trained_on': metadata.get('trained_on'),
    })


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
