# 🏀 NBA Rookie Bust Predictor

A machine learning web app that predicts whether an NBA rookie will become a bust or a successful player based on their rookie season statistics.

![Model Stats](https://img.shields.io/badge/R²-61.7%25-blue)
![Bust Detection](https://img.shields.io/badge/Bust%20F1-74.5%25-green)
![Recall](https://img.shields.io/badge/Bust%20Recall-81.8%25-orange)

## 🎯 What It Does

- **Predicts Win Shares** for Years 2-5 of a player's career
- **Detects Busts** with 81.8% recall (catches most busts)
- **Classifies Players** into tiers: Bust → Below Average → Average Starter → Quality Starter → Star

## 🚀 Quick Start

### 1. Install Requirements

```bash
pip install flask flask-cors pandas numpy scikit-learn
```

### 2. Run the Server

```bash
python app.py
```

You should see:
```
==================================================
 NBA Rookie Prediction API
==================================================
 Model: Ridge Regression
 R²: 0.617
 Bust Detection F1: 0.745
==================================================
 Running on http://localhost:5000
==================================================
```

### 3. Open the Web App

Open `index.html` in your web browser (just double-click it).

### 4. Enter Stats & Predict!

- Enter a rookie's stats from [Basketball Reference](https://www.basketball-reference.com/)
- Or click a preset button (LeBron, Anthony Bennett, etc.)
- Click "Predict Career"

## 📁 Files

| File | Description |
|------|-------------|
| `index.html` | Frontend web interface |
| `app.py` | Flask API server |
| `nba_advanced_model.pkl` | Trained ML model |
| `simple_user_guide.py` | Command-line testing tool |

## 📊 Model Details

### Regression Model (Win Shares Prediction)
- **Algorithm:** Ridge Regression
- **Test R²:** 0.617 (explains 61.7% of variance)
- **Test RMSE:** 7.40 Win Shares
- **Test MAE:** 5.22 Win Shares

### Classification Model (Bust Detection)
- **Algorithm:** AdaBoost Classifier
- **Accuracy:** 73.8%
- **Precision:** 68.4%
- **Recall:** 81.8% (catches 82% of actual busts)
- **F1 Score:** 0.745

### Position-Specific Performance
| Position | R² | RMSE | Bust F1 |
|----------|-----|------|---------|
| Forward | 0.752 | 5.91 | 0.762 |
| Guard | 0.377 | 9.72 | 0.659 |
| Center | 0.194 | 9.07 | 0.571 |

## 🏀 Example Predictions

### Anthony Bennett (#1 Pick 2013) - BUST
```
Predicted Win Shares: 2.6
Bust Probability: 50.6%
Verdict: ⚠️ BUST
Actual: ~1.0 WS (Out of NBA by 25) ✓ CORRECT
```

### LeBron James (2003) - STAR
```
Predicted Win Shares: 38.8
Bust Probability: 27.6%
Verdict: ✅ NOT A BUST
Actual: 64.6 WS (GOAT debate) ✓ CORRECT
```

## 📈 Win Share Tiers

| Tier | Win Shares (Yrs 2-5) | Description |
|------|---------------------|-------------|
| Negative Value | < 0 | Actively hurt their team |
| Bust | 0-5 | Didn't contribute (53% of rookies!) |
| Below Average | 5-15 | Role player at best |
| Average Starter | 15-25 | Solid NBA player |
| Quality Starter | 25-40 | Very good player |
| Star | 40+ | All-Star caliber |

## 🔧 API Endpoints

### POST `/predict`
Predict a player's career from their rookie stats.

**Request:**
```json
{
    "Age": 19,
    "G": 79,
    "GS": 79,
    "MP": 3122,
    "PER": 18.3,
    "WS": 5.1,
    ...
}
```

**Response:**
```json
{
    "success": true,
    "predicted_win_shares": 38.8,
    "bust_probability": 27.6,
    "is_bust": false,
    "tier": "Quality Starter"
}
```

### GET `/health`
Health check endpoint.

## 🛠️ Tech Stack

- **Backend:** Python, Flask, scikit-learn
- **Frontend:** HTML, CSS, JavaScript
- **Models:** Ridge Regression, AdaBoost, Gradient Boosting
- **Data:** Basketball Reference rookie stats (2000-2017)

## 📝 Required Stats

**Basic Stats:**
- Age, G, GS, MP, PTS, TRB, AST, STL, BLK, TOV

**Advanced Stats:**
- PER, TS%, USG%, WS, WS/48, BPM, VORP, OWS, DWS

**Shooting:**
- FG%, 3P%, 2P%, eFG%, FT%, 3PAr, FTr

**Percentages:**
- ORB%, DRB%, TRB%, AST%, STL%, BLK%, TOV%

## 🤝 Contributing

Feel free to open issues or submit PRs!

## 📄 License

MIT License - feel free to use this for your own projects!

---

Made with ❤️ and Machine Learning
