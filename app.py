"""
NBA Rookie Prediction API
==========================
Flask backend that serves the ML model predictions.

To run:
    pip install flask flask-cors
    python app.py

Then open index.html in your browser.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
CORS(app)  # Allow requests from the HTML frontend

# Load the model once when server starts
print("Loading model...")
with open('nba_advanced_model.pkl', 'rb') as f:
    model = pickle.load(f)
print("✅ Model loaded!")


def prepare_player(stats):
    """Takes basic stats and creates all engineered features."""
    s = stats.copy()
    
    # Per-game stats
    s['MP_per_G'] = s.get('MP', 0) / max(s.get('G', 1), 1)
    s['PTS_per_G'] = s.get('PTS', 0) / max(s.get('G', 1), 1)
    s['PTS_per_36'] = s.get('PTS', 0) / max(s.get('MP', 1), 1) * 36
    s['AST_per_G'] = s.get('AST', 0) / max(s.get('G', 1), 1)
    s['TRB_per_G'] = s.get('TRB', 0) / max(s.get('G', 1), 1)
    s['STL_per_G'] = s.get('STL', 0) / max(s.get('G', 1), 1)
    s['BLK_per_G'] = s.get('BLK', 0) / max(s.get('G', 1), 1)
    s['TOV_per_G'] = s.get('TOV', 0) / max(s.get('G', 1), 1)
    
    # Efficiency
    s['AST_to_TOV'] = s.get('AST', 0) / max(s.get('TOV', 1), 1)
    s['STL_plus_BLK'] = s.get('STL%', 0) + s.get('BLK%', 0)
    s['PER_per_USG'] = s.get('PER', 0) / max(s.get('USG%', 1), 1)
    
    # Youth factor
    s['Youth_factor'] = 25 - s.get('Age', 22)
    s['Youth_x_WS'] = s['Youth_factor'] * s.get('WS', 0)
    s['Youth_x_PER'] = s['Youth_factor'] * s.get('PER', 0)
    s['Youth_x_VORP'] = s['Youth_factor'] * s.get('VORP', 0)
    
    # Playing time
    s['GS_ratio'] = s.get('GS', 0) / max(s.get('G', 1), 1)
    s['GS_ratio_x_WS'] = s['GS_ratio'] * s.get('WS', 0)
    s['GS_ratio_x_MP'] = s['GS_ratio'] * s.get('MP', 0)
    
    # Volume
    s['FGA_per_G'] = s.get('PTS', 0) / max(s.get('G', 1), 1) / 2
    s['Shot_volume_per_G'] = s['FGA_per_G'] * 1.2
    s['Usage_opportunity'] = s.get('USG%', 0) * s['MP_per_G']
    
    # Composite
    s['True_shooting_volume'] = s.get('TS%', 0) * s.get('USG%', 0)
    s['eFG_x_volume'] = s.get('eFG%', 0) * s['FGA_per_G'] * s.get('G', 1)
    s['BPM_x_MP'] = s.get('BPM', 0) * s.get('MP', 0)
    s['VORP_per_G'] = s.get('VORP', 0) / max(s.get('G', 1), 1)
    
    # Box score
    s['Box_score_impact'] = (s.get('PTS', 0) + 1.2*s.get('TRB', 0) + 
                            1.5*s.get('AST', 0) + 2*s.get('STL', 0) + 
                            2*s.get('BLK', 0) - s.get('TOV', 0))
    s['Box_impact_per_G'] = s['Box_score_impact'] / max(s.get('G', 1), 1)
    s['Box_impact_per_MP'] = s['Box_score_impact'] / max(s.get('MP', 1), 1) * 100
    
    # Log transforms
    s['log_MP'] = np.log1p(s.get('MP', 0))
    s['log_PTS'] = np.log1p(s.get('PTS', 0))
    s['log_WS_pos'] = np.log1p(max(s.get('WS', 0), 0))
    
    # Interactions
    s['Age_x_MP'] = s.get('Age', 22) * s.get('MP', 0)
    s['PER_x_MP'] = s.get('PER', 0) * s.get('MP', 0)
    s['WS48_x_MP'] = s.get('WS/48', 0) * s.get('MP', 0)
    s['Age_squared'] = s.get('Age', 22) ** 2
    
    # Shooting
    s['Three_point_reliance'] = s.get('3PAr', 0)
    s['FT_rate'] = s.get('FTr', 0)
    
    # Defense/Offense
    s['Defensive_impact'] = s.get('STL%', 0) + s.get('BLK%', 0) + s.get('DRB%', 0) / 10
    s['Defensive_WS_rate'] = s.get('DWS', 0) / max(s.get('MP', 1), 1) * 1000
    s['Offensive_load'] = s.get('USG%', 0) * s.get('AST%', 0) / 100
    s['Scoring_efficiency'] = s.get('PTS', 0) / max(s.get('PTS', 0) / 2 + 0.44 * s.get('PTS', 0) * s.get('FTr', 0), 1)
    
    return s


def get_tier(ws):
    """Get player tier based on predicted Win Shares."""
    if ws < 0:
        return "Negative Value"
    elif ws < 5:
        return "Bust"
    elif ws < 15:
        return "Below Average"
    elif ws < 25:
        return "Average Starter"
    elif ws < 40:
        return "Quality Starter"
    else:
        return "Star"


@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint."""
    try:
        # Get stats from request
        stats = request.json
        
        # Prepare features
        full_stats = prepare_player(stats)
        
        # Create DataFrame with all required features
        feature_cols = model['feature_columns']
        X = pd.DataFrame([full_stats])
        
        for col in feature_cols:
            if col not in X.columns:
                X[col] = 0
        
        X = X[feature_cols].fillna(0)
        
        # Get models
        scaler = model['scaler']
        reg_model = model['best_regression_model']
        clf_model = model['best_classification_model']
        
        # Make predictions
        if model['best_regression_needs_scaling']:
            X_scaled = scaler.transform(X)
            predicted_ws = float(reg_model.predict(X_scaled)[0])
        else:
            predicted_ws = float(reg_model.predict(X)[0])
        
        if model['best_classification_needs_scaling']:
            X_scaled = scaler.transform(X)
            bust_prob = float(clf_model.predict_proba(X_scaled)[0][1])
        else:
            bust_prob = float(clf_model.predict_proba(X)[0][1])
        
        # Return results
        return jsonify({
            'success': True,
            'predicted_win_shares': round(predicted_ws, 1),
            'bust_probability': round(bust_prob * 100, 1),
            'is_bust': bust_prob > 0.5,
            'tier': get_tier(predicted_ws),
            'model_info': {
                'regression_model': model['best_regression_name'],
                'classification_model': model['best_classification_name'],
                'r_squared': round(model['regression_performance']['test_r2'], 3),
                'bust_f1': round(model['classification_performance']['f1_score'], 3)
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'model_loaded': True})


if __name__ == '__main__':
    print("\n" + "="*50)
    print(" NBA Rookie Prediction API")
    print("="*50)
    print(f" Model: {model['best_regression_name']}")
    print(f" R²: {model['regression_performance']['test_r2']:.3f}")
    print(f" Bust Detection F1: {model['classification_performance']['f1_score']:.3f}")
    print("="*50)
    print(" Running on http://localhost:5000")
    print("="*50 + "\n")
    
import os
port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)

