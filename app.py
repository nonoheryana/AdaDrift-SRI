"""
=============================================================================
AdaDrift-SRI: Interactive Web Dashboard
Flask-based GUI for Adaptive Drift-Aware Online Learning
IEEE Conference Demo Version — FIXED
=============================================================================
"""
import os
import json
import time
import io
import base64
import traceback
from datetime import datetime
from threading import Thread

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from math import pi

from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for, flash
from werkzeug.utils import secure_filename
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
from river import tree as rtree, preprocessing as rprep
import plotly.graph_objs as go
import plotly.utils
import plotly.express as px

# ═══════════════════════════════════════════════════════════════════════════
# FLASK APP INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════
app = Flask(__name__)
app.secret_key = 'adadrift-sri-ieee-2024-secret-key'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

training_progress = {
    'status': 'idle',
    'progress': 0,
    'message': '',
    'results': None,
    'error': None
}

# ═══════════════════════════════════════════════════════════════════════════
# ALGORITHM CLASSES (FIXED)
# ═══════════════════════════════════════════════════════════════════════════

class PHDetector:
    """C1: Page-Hinkley Drift Detector"""
    def __init__(self, delta=0.005, lam=40., alpha=0.9999):
        self.delta = delta
        self.lam = lam
        self.alpha = alpha
        self.sum_ = 0.
        self.mu = 0.
    
    def update(self, x):
        self.mu = self.alpha * self.mu + (1 - self.alpha) * x
        self.sum_ = max(0., self.sum_ + x - self.mu - self.delta)
        return self.sum_ > self.lam
    
    def reset(self):
        self.sum_ = 0.


class AdaDriftSRI:
    """AdaDrift-SRI: Adaptive Drift-Aware Online Learning Algorithm"""
    def __init__(self, lam=40., delta=0.005, win=60):
        self.ph = PHDetector(delta=delta, lam=lam)
        self.win = win
        self._make()
        self.rsc = rprep.StandardScaler()
        self.n = 0
        self.n_drifts = 0
        self.drift_idx = []
        self.drift_dates = []
        self.buf_t = []
        self.buf_p = []
        self.rmse_h = []
        self.mae_h = []
        self.preds = []
        # FIX: Simpan semua y_true untuk perhitungan cumulative R²
        self.y_true_history = []
    
    def _make(self):
        self.model = rtree.HoeffdingAdaptiveTreeRegressor(
            max_depth=10, delta=1e-7, tau=0.05,
            leaf_prediction='adaptive', grace_period=50, seed=42
        )
    
    def _d(self, xi):
        return {f'f{i}': float(v) for i, v in enumerate(xi)}
    
    def warm_start(self, X_sc, y):
        for xi, yi in zip(X_sc, y):
            xd = self._d(xi)
            self.rsc.learn_one(xd)
            self.model.learn_one(self.rsc.transform_one(xd), yi)
    
    def step(self, xi_sc, yi, date=None):
        self.n += 1
        xd = self._d(xi_sc)
        self.rsc.learn_one(xd)
        xs = self.rsc.transform_one(xd)
        
        # Predict
        yp = self.model.predict_one(xs)
        yp = yp if yp is not None else 0.
        
        # Drift detection
        fired = self.ph.update(abs(yi - yp))
        if fired and self.n > 30:
            self.n_drifts += 1
            self.drift_idx.append(self.n)
            if date is not None:
                self.drift_dates.append(str(pd.Timestamp(date).date()))
            self.ph.reset()
            self._make()
        
        # Learn
        self.model.learn_one(xs, yi)
        
        # Update buffers
        self.buf_t.append(yi)
        self.buf_p.append(yp)
        if len(self.buf_t) > self.win:
            self.buf_t.pop(0)
            self.buf_p.pop(0)
        
        # FIX: Simpan y_true untuk cumulative R²
        self.y_true_history.append(yi)
        self.preds.append(yp)
        
        # Rolling RMSE
        rmse_w = (np.sqrt(np.mean((np.array(self.buf_t) - np.array(self.buf_p))**2))
                  if len(self.buf_t) >= 10 else float('nan'))
        mae_w = (np.mean(np.abs(np.array(self.buf_t) - np.array(self.buf_p)))
                 if len(self.buf_t) >= 10 else float('nan'))
        
        self.rmse_h.append(rmse_w)
        self.mae_h.append(mae_w)
        
        return yp, fired
    
    def get_cumulative_r2(self):
        """Hitung cumulative R² dengan aman"""
        if self.n < 10:
            return []
        
        cum_r2 = []
        for i in range(10, self.n + 1):
            try:
                r2_val = r2_score(
                    np.array(self.y_true_history[:i]),
                    np.array(self.preds[:i])
                )
                cum_r2.append(float(r2_val))
            except:
                cum_r2.append(0.0)
        return cum_r2


class SWRT:
    """B3: Sliding-Window Retrain XGBoost (Fixed Scaler — No Leakage)"""
    def __init__(self, every=50, win=500):
        self.every = every
        self.win = win
        self.Xb = []
        self.yb = []
        self.model = None
        self.n = 0
        self.sc_mean = None
        self.sc_std = None
    
    def warm_start(self, X, y):
        self.Xb = list(X[-self.win:])
        self.yb = list(y[-self.win:])
        X_arr = np.array(self.Xb)
        self.sc_mean = X_arr.mean(axis=0)
        self.sc_std = X_arr.std(axis=0)
        self.sc_std[self.sc_std == 0] = 1e-8
    
    def _transform(self, X):
        return (X - self.sc_mean) / self.sc_std
    
    def step(self, xi, yi):
        self.n += 1
        self.Xb.append(xi.copy())
        self.yb.append(yi)
        if len(self.Xb) > self.win:
            self.Xb.pop(0)
            self.yb.pop(0)
        
        xi_scaled = self._transform(xi.reshape(1, -1))
        
        if self.n % self.every == 0 and len(self.Xb) >= 50:
            Xb = np.array(self.Xb)
            yb = np.array(self.yb)
            Xb_scaled = self._transform(Xb)
            self.model = xgb.XGBRegressor(
                n_estimators=80, max_depth=4,
                learning_rate=0.1, verbosity=0, n_jobs=-1, random_state=42
            )
            self.model.fit(Xb_scaled, yb)
        
        if self.model is None:
            return 0.
        return float(self.model.predict(xi_scaled)[0])


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def create_plotly_timeline(dates, y_true, predictions_dict, crisis_mask, drift_dates_list):
    """Create interactive Plotly timeline chart"""
    # FIX: Convert to numpy array for min/max operations
    y_true_arr = np.array(y_true) if not isinstance(y_true, np.ndarray) else y_true
    y_min = float(np.min(y_true_arr))
    y_max = float(np.max(y_true_arr))
    
    traces = []
    
    # 1. True values
    traces.append(go.Scatter(
        x=dates, 
        y=y_true_arr.tolist() if isinstance(y_true_arr, np.ndarray) else y_true,
        mode='lines',
        name='Actual SRI',
        line=dict(color='black', width=1.5),
        opacity=0.7
    ))
    
    # 2. Predictions for each model
    colors = {
        'AdaDrift-SRI': '#1f77b4',
        'Static-XGB': '#ff7f0e',
        'Online-SGD': '#2ca02c',
        'SWRT-XGB': '#d62728'
    }
    
    for model_name, preds in predictions_dict.items():
        traces.append(go.Scatter(
            x=dates, 
            y=preds,
            mode='lines',
            name=model_name,
            line=dict(color=colors.get(model_name, '#999999'), width=1.2, dash='dash'),
            opacity=0.8
        ))
    
    # 3. Crisis shading (FIX: handle list properly)
    if crisis_mask is not None:
        crisis_mask_arr = np.array(crisis_mask) if not isinstance(crisis_mask, np.ndarray) else crisis_mask
        
        if np.any(crisis_mask_arr):
            # Find crisis regions
            in_crisis = False
            crisis_start = None
            
            for i in range(len(crisis_mask_arr)):
                is_crisis = bool(crisis_mask_arr[i])
                
                if is_crisis and not in_crisis:
                    crisis_start = dates[i]
                    in_crisis = True
                elif not is_crisis and in_crisis:
                    # End of crisis region
                    if crisis_start is not None:
                        traces.append(go.Scatter(
                            x=[crisis_start, dates[i-1], dates[i-1], crisis_start],
                            y=[y_min, y_min, y_max, y_max],
                            fill='toself',
                            fillcolor='rgba(148, 103, 189, 0.15)',
                            line=dict(width=0),
                            name='Crisis',
                            showlegend=(i == 0),  # Only show legend once
                            hoverinfo='skip'
                        ))
                    in_crisis = False
                    crisis_start = None
            
            # Handle if still in crisis at end
            if in_crisis and crisis_start is not None:
                traces.append(go.Scatter(
                    x=[crisis_start, dates[-1], dates[-1], crisis_start],
                    y=[y_min, y_min, y_max, y_max],
                    fill='toself',
                    fillcolor='rgba(148, 103, 189, 0.15)',
                    line=dict(width=0),
                    name='Crisis',
                    showlegend=True,
                    hoverinfo='skip'
                ))
    
    # 4. Drift markers
    if drift_dates_list and len(drift_dates_list) > 0:
        drift_x = []
        drift_y = []
        
        for d in drift_dates_list:
            d_str = str(d)
            if d_str in dates:
                idx = dates.index(d_str)
                drift_x.append(d_str)
                drift_y.append(y_true_arr[idx] if isinstance(y_true_arr, np.ndarray) else y_true[idx])
        
        if drift_x:
            traces.append(go.Scatter(
                x=drift_x,
                y=drift_y,
                mode='markers',
                name='Drift Events',
                marker=dict(
                    color='#e377c2',
                    size=12,
                    symbol='triangle-down',
                    line=dict(width=2, color='darkred')
                ),
                hovertext=[f'Drift: {d}' for d in drift_x],
                hoverinfo='text'
            ))
    
    # Layout
    layout = go.Layout(
        title='Systemic Risk Index: Actual vs Predictions',
        xaxis=dict(title='Date', gridcolor='#eeeeee', showgrid=True),
        yaxis=dict(title='SRI Value', gridcolor='#eeeeee', showgrid=True),
        hovermode='x unified',
        legend=dict(
            orientation='h', 
            y=1.15, 
            x=0.5, 
            xanchor='center',
            font=dict(size=11)
        ),
        template='plotly_white',
        margin=dict(l=60, r=30, t=80, b=60),
        height=500
    )
    
    fig = go.Figure(data=traces, layout=layout)
    return fig


def run_training(filepath, warmup_pct, ph_lambda, ph_delta, sw_window):
    """Run AdaDrift-SRI training pipeline"""
    global training_progress
    
    try:
        training_progress['status'] = 'running'
        training_progress['progress'] = 0
        training_progress['message'] = 'Loading data...'
        
        # Load data
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        banks = ['BBRI', 'BBTN', 'BMRI', 'BBNI']
        
        # Build features
        FEATURES = []
        for b in banks:
            FEATURES += [f'{b}_Volatility', f'{b}_Return', f'{b}_LogReturn',
                         f'{b}_BB_Width', f'{b}_ROC_10', f'{b}_Momentum_10']
        FEATURES += ['Avg_Volatility', 'Avg_Correlation', 'Max_Correlation',
                     'Avg_Correlation_lag1', 'Avg_Correlation_lag2', 'Avg_Correlation_lag3',
                     'Crisis_Intensity_30d']
        for b in banks:
            for k in range(1, 3):
                FEATURES.append(f'{b}_Return_lag{k}')
        
        # Filter features yang ada di dataset
        FEATURES = [f for f in FEATURES if f in df.columns]
        print(f"[INFO] Using {len(FEATURES)} features: {FEATURES[:5]}...")
        
        TARGET = 'Systemic_Risk_Index'
        if TARGET not in df.columns:
            raise ValueError(f"Target column '{TARGET}' not found. Available columns: {list(df.columns)[:10]}...")
        
        # Prepare data
        required_cols = FEATURES + [TARGET, 'Crisis_Period', 'Date', 'Year']
        available_cols = [c for c in required_cols if c in df.columns]
        df_c = df[available_cols].dropna().reset_index(drop=True)
        
        N = len(df_c)
        WARMUP = int(N * warmup_pct)
        
        X_all = df_c[FEATURES].values
        y_all = df_c[TARGET].values
        dates = df_c['Date'].values
        crisis = df_c['Crisis_Period'].values
        years = df_c['Year'].values
        
        # Split
        X_wu = X_all[:WARMUP]
        y_wu = y_all[:WARMUP]
        X_ev = X_all[WARMUP:]
        y_ev = y_all[WARMUP:]
        cr_ev = crisis[WARMUP:]
        dt_ev = dates[WARMUP:]
        yr_ev = years[WARMUP:]
        
        training_progress['progress'] = 5
        training_progress['message'] = f'Data loaded: {N} obs, {len(FEATURES)} features'
        
        # Scale on warm-up only
        sc = StandardScaler()
        sc.fit(X_wu)
        X_wu_sc = sc.transform(X_wu)
        X_ev_sc = sc.transform(X_ev)
        
        training_progress['progress'] = 10
        training_progress['message'] = 'Training models...'
        
        # B1: Static-XGB
        xgb_b1 = xgb.XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            random_state=42, verbosity=0, n_jobs=-1
        )
        xgb_b1.fit(X_wu_sc, y_wu)
        
        # B2: Online-SGD
        sgd = SGDRegressor(
            loss='huber', epsilon=0.01, alpha=0.0001,
            learning_rate='adaptive', eta0=0.01, max_iter=1, tol=None, random_state=42
        )
        sgd.partial_fit(X_wu_sc, y_wu)
        
        # B3: SWRT-XGB
        swrt = SWRT(every=50, win=500)
        swrt.warm_start(X_wu_sc, y_wu)
        
        # AdaDrift-SRI
        ada = AdaDriftSRI(lam=ph_lambda, delta=ph_delta, win=sw_window)
        ada.warm_start(X_wu_sc, y_wu)
        
        training_progress['progress'] = 20
        training_progress['message'] = 'Running prequential evaluation...'
        
        # Prequential loop
        t0 = time.time()
        b1p = []
        b2p = []
        b3p = []
        dflag = []
        n_ev = len(X_ev)
        
        for i, (xi_sc, yi) in enumerate(zip(X_ev_sc, y_ev)):
            if i % 200 == 0:
                progress = 20 + int((i / n_ev) * 70)
                training_progress['progress'] = min(progress, 90)
                training_progress['message'] = f'Processing: {i}/{n_ev} ({i/n_ev*100:.0f}%)'
            
            # AdaDrift-SRI
            yp, fired = ada.step(xi_sc, yi, dt_ev[i])
            dflag.append(fired)
            
            # Static-XGB (no update)
            b1p.append(float(xgb_b1.predict(xi_sc.reshape(1, -1))[0]))
            
            # Online-SGD (update after predict)
            b2p.append(float(sgd.predict(xi_sc.reshape(1, -1))[0]))
            sgd.partial_fit(xi_sc.reshape(1, -1), np.array([yi]))
            
            # SWRT-XGB
            b3p.append(swrt.step(xi_sc, yi))
        
        elapsed = time.time() - t0
        
        training_progress['progress'] = 95
        training_progress['message'] = 'Computing metrics...'
        
        # Convert to arrays
        ada_preds_arr = np.array(ada.preds)
        b1p_arr = np.array(b1p)
        b2p_arr = np.array(b2p)
        b3p_arr = np.array(b3p)
        
        # Compute metrics
        def compute_metrics(y_true, y_pred, crisis_mask):
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            
            r2c = r2_score(y_true[crisis_mask], y_pred[crisis_mask]) if crisis_mask.sum() > 0 else float('nan')
            r2n = r2_score(y_true[~crisis_mask], y_pred[~crisis_mask]) if (~crisis_mask).sum() > 0 else float('nan')
            
            return {
                'R2_Overall': round(r2, 4),
                'RMSE_Overall': round(rmse, 4),
                'MAE_Overall': round(mae, 4),
                'R2_Crisis': round(r2c, 4) if not np.isnan(r2c) else 'N/A',
                'R2_NonCrisis': round(r2n, 4) if not np.isnan(r2n) else 'N/A',
                'Crisis_Delta': round(r2c - r2, 4) if not np.isnan(r2c) else 'N/A'
            }
        
        crisis_mask = cr_ev == 1
        
        results = {
            'AdaDrift-SRI': compute_metrics(y_ev, ada_preds_arr, crisis_mask),
            'Static-XGB': compute_metrics(y_ev, b1p_arr, crisis_mask),
            'Online-SGD': compute_metrics(y_ev, b2p_arr, crisis_mask),
            'SWRT-XGB': compute_metrics(y_ev, b3p_arr, crisis_mask)
        }
        
        results['AdaDrift-SRI']['Drifts_Detected'] = ada.n_drifts
        results['AdaDrift-SRI']['Time_per_Obs_ms'] = round(elapsed / n_ev * 1000, 2)
        
        # Drift events table
        drift_events = []
        if ada.drift_idx:
            for idx, drift_idx in enumerate(ada.drift_idx):
                actual_idx = min(drift_idx, len(y_ev)-1)
                actual_val = float(y_ev[actual_idx])
                pred_val = float(ada_preds_arr[actual_idx])
                error_val = abs(actual_val - pred_val)
                is_crisis = bool(cr_ev[actual_idx])
                drift_date = ada.drift_dates[idx] if idx < len(ada.drift_dates) else str(dt_ev[actual_idx])
                
                drift_events.append({
                    'Drift_Number': idx + 1,
                    'Observation_Index': drift_idx,
                    'Date': str(drift_date),
                    'Actual_SRI': round(actual_val, 4),
                    'Predicted_SRI': round(pred_val, 4),
                    'Absolute_Error': round(error_val, 4),
                    'During_Crisis': is_crisis
                })
        
        # Dataset info
        dataset_info = {
            'Total_Observations': int(N),
            'Warmup_Size': int(WARMUP),
            'Warmup_Pct': round(warmup_pct * 100, 1),
            'Eval_Size': int(N - WARMUP),
            'Features_Count': len(FEATURES),
            'Crisis_Observations': int(crisis_mask.sum()),
            'Crisis_Pct': round(crisis_mask.sum() / len(crisis_mask) * 100, 1),
            'Date_Range_Warmup': f"{pd.Timestamp(dates[0]).date()} to {pd.Timestamp(dates[WARMUP-1]).date()}",
            'Date_Range_Eval': f"{pd.Timestamp(dates[WARMUP]).date()} to {pd.Timestamp(dates[-1]).date()}"
        }
        
        # Plot data (FIX: ensure proper types for JSON serialization)
        plot_data = {
            'dates': [str(d) for d in dt_ev],
            'y_true': [float(x) for x in y_ev],
            'crisis': [bool(x) for x in cr_ev],
            'predictions': {
                'AdaDrift-SRI': [float(x) for x in ada_preds_arr],
                'Static-XGB': [float(x) for x in b1p_arr],
                'Online-SGD': [float(x) for x in b2p_arr],
                'SWRT-XGB': [float(x) for x in b3p_arr]
            },
            'drift_dates': [str(d) for d in ada.drift_dates] if ada.drift_dates else [],
            'rolling_rmse': [float(x) if x is not None and not np.isnan(x) else None for x in ada.rmse_h]
        }
        
        training_progress['status'] = 'completed'
        training_progress['progress'] = 100
        training_progress['message'] = f'Training completed in {elapsed:.1f}s!'
        training_progress['results'] = {
            'metrics': results,
            'drift_events': drift_events,
            'dataset_info': dataset_info,
            'plot_data': plot_data,
            'elapsed_time': round(elapsed, 1),
            'n_drifts': int(ada.n_drifts)
        }
        
    except Exception as e:
        training_progress['status'] = 'error'
        training_progress['error'] = str(e)
        training_progress['message'] = f'Error: {str(e)}'
        traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            warmup_pct = float(request.form.get('warmup_pct', 0.30))
            ph_lambda = float(request.form.get('ph_lambda', 40))
            ph_delta = float(request.form.get('ph_delta', 0.005))
            sw_window = int(request.form.get('sw_window', 60))
            
            global training_progress
            training_progress = {
                'status': 'idle',
                'progress': 0,
                'message': '',
                'results': None,
                'error': None
            }
            
            thread = Thread(target=run_training, args=(filepath, warmup_pct, ph_lambda, ph_delta, sw_window))
            thread.daemon = True
            thread.start()
            
            return redirect(url_for('training'))
        else:
            flash('Please upload a CSV file', 'error')
    
    return render_template('upload.html')

@app.route('/training')
def training():
    return render_template('training.html')

@app.route('/results')
def results():
    if training_progress.get('status') != 'completed' or training_progress.get('results') is None:
        flash('No results available. Please upload and train a model first.', 'warning')
        return redirect(url_for('upload'))
    return render_template('results.html', results=training_progress['results'])

@app.route('/drift_analysis')
def drift_analysis():
    if training_progress.get('status') != 'completed' or training_progress.get('results') is None:
        flash('No results available. Please upload and train a model first.', 'warning')
        return redirect(url_for('upload'))
    return render_template('drift_analysis.html', results=training_progress['results'])

@app.route('/api/progress')
def api_progress():
    return jsonify({
        'status': training_progress.get('status', 'idle'),
        'progress': training_progress.get('progress', 0),
        'message': training_progress.get('message', ''),
        'error': training_progress.get('error', None)
    })

@app.route('/api/results')
def api_results():
    if training_progress.get('results') is None:
        return jsonify({'error': 'No results available'}), 404
    return jsonify(training_progress['results'])

@app.route('/api/generate_plot')
def api_generate_plot():
    """Generate and return Plotly figure as JSON"""
    if training_progress.get('results') is None:
        return jsonify({'error': 'No data available'}), 404
    
    try:
        plot_data = training_progress['results']['plot_data']
        plot_type = request.args.get('type', 'timeline')
        
        if plot_type == 'timeline':
            fig = create_plotly_timeline(
                plot_data['dates'],
                plot_data['y_true'],
                plot_data['predictions'],
                plot_data['crisis'],
                plot_data.get('drift_dates', [])
            )
            # Convert Plotly figure to JSON
            fig_json = fig.to_json()
            return jsonify(json.loads(fig_json))
        
        return jsonify({'error': 'Invalid plot type'}), 400
    
    except Exception as e:
        # Return error details for debugging
        traceback.print_exc()
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/download/<filetype>')
def api_download(filetype):
    if training_progress.get('results') is None:
        return jsonify({'error': 'No results available'}), 404
    
    results = training_progress['results']
    
    if filetype == 'metrics':
        df = pd.DataFrame(results['metrics']).T
        df.index.name = 'Algorithm'
    elif filetype == 'drifts':
        df = pd.DataFrame(results['drift_events'])
    elif filetype == 'predictions':
        df = pd.DataFrame({
            'Date': results['plot_data']['dates'],
            'Actual_SRI': results['plot_data']['y_true'],
            'AdaDrift_SRI': results['plot_data']['predictions']['AdaDrift-SRI'],
            'Static_XGB': results['plot_data']['predictions']['Static-XGB'],
            'Online_SGD': results['plot_data']['predictions']['Online-SGD'],
            'SWRT_XGB': results['plot_data']['predictions']['SWRT-XGB'],
            'Crisis': results['plot_data']['crisis']
        })
    else:
        return jsonify({'error': 'Invalid file type'}), 400
    
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{filetype}.csv')
    df.to_csv(output_path, index=(filetype == 'metrics'))
    return send_file(output_path, as_attachment=True, download_name=f'{filetype}.csv')


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("="*60)
    print("AdaDrift-SRI Web Dashboard")
    print("http://localhost:5000")
    print("="*60)
    app.run(debug=True, host='0.0.0.0', port=5000)