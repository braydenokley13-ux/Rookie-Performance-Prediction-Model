 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/app.py b/app.py
index ef409cf2af2d69d7b59821d5cc8407e196b7f9f9..142b10c8a9adf58700ea40fc7908636353560016 100644
--- a/app.py
+++ b/app.py
@@ -1,56 +1,95 @@
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
+from sklearn.dummy import DummyClassifier, DummyRegressor
+from sklearn.preprocessing import StandardScaler
 
 app = Flask(__name__)
 CORS(app)  # Allow requests from the HTML frontend
 
-# Load the model once when server starts
-print("Loading model...")
-with open('nba_advanced_model.pkl', 'rb') as f:
-    model = pickle.load(f)
-print("✅ Model loaded!")
+BASE_FEATURES = [
+    'Age', 'G', 'GS', 'MP', 'PTS', 'TRB', 'AST', 'STL', 'BLK', 'TOV',
+    'PER', 'TS%', 'USG%', 'WS', 'WS/48', 'BPM', 'VORP', 'OWS', 'DWS',
+    'OBPM', 'DBPM', 'FG%', '3P%', '2P%', 'eFG%', 'FT%', '3PAr', 'FTr',
+    'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%'
+]
+
+
+def build_fallback_model(feature_columns):
+    """Create a lightweight fallback model so the API can boot without a pickle."""
+    n_features = len(feature_columns)
+    X_dummy = np.zeros((2, n_features))
+
+    reg_model = DummyRegressor(strategy='constant', constant=0.0)
+    reg_model.fit(X_dummy, [0.0, 0.0])
+
+    clf_model = DummyClassifier(strategy='prior')
+    clf_model.fit(X_dummy, [0, 1])
+
+    scaler = StandardScaler()
+    scaler.fit(X_dummy)
+
+    return {
+        'feature_columns': feature_columns,
+        'scaler': scaler,
+        'best_regression_model': reg_model,
+        'best_classification_model': clf_model,
+        'best_regression_needs_scaling': False,
+        'best_classification_needs_scaling': False,
+        'best_regression_name': 'DummyRegressor (fallback)',
+        'best_classification_name': 'DummyClassifier (fallback)',
+        'regression_performance': {'test_r2': 0.0},
+        'classification_performance': {'f1_score': 0.0},
+        'fallback': True,
+    }
+
+
+def base_feature_template():
+    """Provide all expected base features with zero defaults."""
+    return {feature: 0 for feature in BASE_FEATURES}
 
 
 def prepare_player(stats):
     """Takes basic stats and creates all engineered features."""
     s = stats.copy()
+    for feature, default_value in base_feature_template().items():
+        s.setdefault(feature, default_value)
     
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
@@ -76,50 +115,63 @@ def prepare_player(stats):
     
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
 
 
+# Load the model once when server starts
+print("Loading model...")
+try:
+    with open('nba_advanced_model.pkl', 'rb') as f:
+        model = pickle.load(f)
+    print("✅ Model loaded!")
+except Exception as exc:
+    print(f"⚠️ Failed to load model pickle: {exc}")
+    print("⚠️ Falling back to a dummy model. Predictions will be placeholders.")
+    fallback_stats = prepare_player(base_feature_template())
+    model = build_fallback_model(sorted(fallback_stats.keys()))
+
+
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
@@ -155,43 +207,46 @@ def predict():
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
-    return jsonify({'status': 'ok', 'model_loaded': True})
+    return jsonify({
+        'status': 'ok',
+        'model_loaded': True,
+        'fallback_model': model.get('fallback', False)
+    })
 
 
 if __name__ == '__main__':
     print("\n" + "="*50)
     print(" NBA Rookie Prediction API")
     print("="*50)
-    print(f" Model: {model['best_regression_name']}")
-    print(f" R²: {model['regression_performance']['test_r2']:.3f}")
-    print(f" Bust Detection F1: {model['classification_performance']['f1_score']:.3f}")
+    print(f" Model: {model.get('best_regression_name', 'Unknown')}")
+    print(f" R²: {model.get('regression_performance', {}).get('test_r2', 0.0):.3f}")
+    print(f" Bust Detection F1: {model.get('classification_performance', {}).get('f1_score', 0.0):.3f}")
     print("="*50)
     print(" Running on http://localhost:5000")
     print("="*50 + "\n")
     
 import os
 port = int(os.environ.get("PORT", 5000))
 app.run(host="0.0.0.0", port=port)
-
 
EOF
)
