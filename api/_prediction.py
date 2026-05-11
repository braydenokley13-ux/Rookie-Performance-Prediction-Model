"""Shared prediction logic used by Vercel serverless functions."""

import os
import pickle

import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.preprocessing import StandardScaler

BASE_FEATURES = [
    'Age', 'G', 'GS', 'MP', 'PTS', 'TRB', 'AST', 'STL', 'BLK', 'TOV',
    'PER', 'TS%', 'USG%', 'WS', 'WS/48', 'BPM', 'VORP', 'OWS', 'DWS',
    'OBPM', 'DBPM', 'FG%', '3P%', '2P%', 'eFG%', 'FT%', '3PAr', 'FTr',
    'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%'
]


def base_feature_template():
    return {feature: 0 for feature in BASE_FEATURES}


def prepare_player(stats):
    s = stats.copy()
    for feature, default_value in base_feature_template().items():
        s.setdefault(feature, default_value)

    s['MP_per_G'] = s.get('MP', 0) / max(s.get('G', 1), 1)
    s['PTS_per_G'] = s.get('PTS', 0) / max(s.get('G', 1), 1)
    s['PTS_per_36'] = s.get('PTS', 0) / max(s.get('MP', 1), 1) * 36
    s['AST_per_G'] = s.get('AST', 0) / max(s.get('G', 1), 1)
    s['TRB_per_G'] = s.get('TRB', 0) / max(s.get('G', 1), 1)
    s['STL_per_G'] = s.get('STL', 0) / max(s.get('G', 1), 1)
    s['BLK_per_G'] = s.get('BLK', 0) / max(s.get('G', 1), 1)
    s['TOV_per_G'] = s.get('TOV', 0) / max(s.get('G', 1), 1)

    s['AST_to_TOV'] = s.get('AST', 0) / max(s.get('TOV', 1), 1)
    s['STL_plus_BLK'] = s.get('STL%', 0) + s.get('BLK%', 0)
    s['PER_per_USG'] = s.get('PER', 0) / max(s.get('USG%', 1), 1)

    s['Youth_factor'] = 25 - s.get('Age', 22)
    s['Youth_x_WS'] = s['Youth_factor'] * s.get('WS', 0)
    s['Youth_x_PER'] = s['Youth_factor'] * s.get('PER', 0)
    s['Youth_x_VORP'] = s['Youth_factor'] * s.get('VORP', 0)

    s['GS_ratio'] = s.get('GS', 0) / max(s.get('G', 1), 1)
    s['GS_ratio_x_WS'] = s['GS_ratio'] * s.get('WS', 0)
    s['MP_x_PER'] = s.get('MP', 0) * s.get('PER', 0)

    s['log_MP'] = np.log1p(s.get('MP', 0))
    s['log_PTS'] = np.log1p(s.get('PTS', 0))
    s['log_WS_pos'] = np.log1p(max(s.get('WS', 0), 0))

    s['Age_x_MP'] = s.get('Age', 22) * s.get('MP', 0)
    s['PER_x_MP'] = s.get('PER', 0) * s.get('MP', 0)
    s['WS48_x_MP'] = s.get('WS/48', 0) * s.get('MP', 0)
    s['Age_squared'] = s.get('Age', 22) ** 2

    s['Three_point_reliance'] = s.get('3PAr', 0)
    s['FT_rate'] = s.get('FTr', 0)

    s['Defensive_impact'] = s.get('STL%', 0) + s.get('BLK%', 0) + s.get('DRB%', 0) / 10
    s['Defensive_WS_rate'] = s.get('DWS', 0) / max(s.get('MP', 1), 1) * 1000
    s['Offensive_load'] = s.get('USG%', 0) * s.get('AST%', 0) / 100
    s['Scoring_efficiency'] = s.get('PTS', 0) / max(
        s.get('PTS', 0) / 2 + 0.44 * s.get('PTS', 0) * s.get('FTr', 0), 1
    )

    return s


def build_fallback_model(feature_columns):
    n_features = len(feature_columns)
    X_dummy = np.zeros((2, n_features))

    reg_model = DummyRegressor(strategy='constant', constant=0.0)
    reg_model.fit(X_dummy, [0.0, 0.0])

    clf_model = DummyClassifier(strategy='prior')
    clf_model.fit(X_dummy, [0, 1])

    scaler = StandardScaler()
    scaler.fit(X_dummy)

    return {
        'feature_columns': feature_columns,
        'scaler': scaler,
        'best_regression_model': reg_model,
        'best_classification_model': clf_model,
        'best_regression_needs_scaling': False,
        'best_classification_needs_scaling': False,
        'best_regression_name': 'DummyRegressor (fallback)',
        'best_classification_name': 'DummyClassifier (fallback)',
        'regression_performance': {'test_r2': 0.0},
        'classification_performance': {'f1_score': 0.0},
        'fallback': True,
    }


_MODEL_FILENAME = 'nba_advanced_model.pkl'
_CANDIDATE_DIRS = [
    os.path.dirname(__file__),
    os.path.dirname(os.path.dirname(__file__)),
    os.getcwd(),
]


def _load_model():
    for directory in _CANDIDATE_DIRS:
        candidate = os.path.join(directory, _MODEL_FILENAME)
        if os.path.exists(candidate):
            try:
                with open(candidate, 'rb') as f:
                    return pickle.load(f)
            except Exception:  # noqa: BLE001 -- fall through to dummy model
                break
    fallback_stats = prepare_player(base_feature_template())
    return build_fallback_model(sorted(fallback_stats.keys()))


_model = None


def get_model():
    global _model
    if _model is None:
        _model = _load_model()
    return _model


def get_tier(ws):
    if ws < 0:
        return "Negative Value"
    if ws < 5:
        return "Bust"
    if ws < 15:
        return "Below Average"
    if ws < 25:
        return "Average Starter"
    if ws < 40:
        return "Quality Starter"
    return "Star"


def run_prediction(stats):
    model = get_model()
    full_stats = prepare_player(stats)

    feature_columns = model['feature_columns']
    X = np.array([[full_stats.get(col, 0) for col in feature_columns]])

    scaler = model['scaler']
    X_scaled = scaler.transform(X)

    reg_model = model['best_regression_model']
    reg_input = X_scaled if model.get('best_regression_needs_scaling') else X
    predicted_ws = float(reg_model.predict(reg_input)[0])

    clf_model = model['best_classification_model']
    clf_input = X_scaled if model.get('best_classification_needs_scaling') else X
    if hasattr(clf_model, 'predict_proba'):
        bust_prob = float(clf_model.predict_proba(clf_input)[0][1])
    else:
        bust_prob = float(clf_model.predict(clf_input)[0])

    return {
        'success': True,
        'predicted_win_shares': round(predicted_ws, 1),
        'bust_probability': round(bust_prob * 100, 1),
        'is_bust': bust_prob > 0.5,
        'tier': get_tier(predicted_ws),
        'model_info': {
            'regression_model': model.get('best_regression_name', 'Unknown'),
            'classification_model': model.get('best_classification_name', 'Unknown'),
            'r_squared': round(model.get('regression_performance', {}).get('test_r2', 0.0), 3),
            'bust_f1': round(model.get('classification_performance', {}).get('f1_score', 0.0), 3),
            'fallback': bool(model.get('fallback', False)),
        },
    }
