# Sports Sponsorship ROI Calculator

Predict **media value**, **brand lift**, and **ROI** for sports sponsorship deals using XGBoost with SHAP explainability.

![Screenshot Placeholder](screenshot.png)

## What This Demonstrates

- **Financial Modeling** — Quantitative ROI analysis across $50K-$50M sponsorship deals with multi-target prediction
- **ML Pipeline** — End-to-end workflow: synthetic data generation, feature engineering, XGBoost training, SHAP interpretation
- **Sports Domain Expertise** — Realistic sponsorship economics calibrated to IEG/Nielsen Sports benchmarks across 8 leagues, 10 markets, and 6 deal types
- **Interactive Data Product** — Production-quality Streamlit app with real-time predictions and visual explainability

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (generate data → train model → launch app)
bash run.sh

# Or, if data/model already exist, just launch the app:
bash run.sh --app
```

The app will open at [http://localhost:8501](http://localhost:8501).

## Architecture

```
sponsorship-roi-calculator/
├── app.py                      # Streamlit web application
├── data/
│   ├── generate_data.py        # Synthetic data generator (5,000 deals)
│   └── sponsorship_deals.csv   # Generated dataset (after running)
├── model/
│   ├── train.py                # XGBoost training pipeline
│   ├── trained_model.joblib    # Saved model artifacts (after training)
│   └── metrics.json            # Evaluation metrics (after training)
├── run.sh                      # One-command pipeline script
├── requirements.txt            # Python dependencies
└── README.md
```

### Data Generation
Generates 5,000 synthetic sponsorship deals with realistic correlations:
- **Features**: sport, market, deal type, annual spend ($50K-$50M log-normal), deal length, audience reach, social following, brand category, 5 activation channel flags
- **Targets**: media value ratio (2-3x avg), brand lift % (3-8% range), ROI score (1-100 composite)
- Correlations calibrated to industry benchmarks (NFL/NBA command higher media value, experiential+social combos drive brand lift)

### Model Training
- Log-transforms for high-variance features (spend, reach, social following)
- Interaction features (sport x market) capture local market dynamics
- XGBoost regressors with regularization (L1/L2, subsampling)
- 80/20 train/test split with MAE, R², and MAPE evaluation

### Streamlit App
- **Sidebar**: Full deal parameter configuration with log-scale spend slider
- **Metric Cards**: Predicted media value ratio, brand lift %, and ROI score
- **SHAP Waterfall**: Feature-level contribution chart showing what drives each prediction
- **Comparable Deals**: Table of similar historical deals for benchmarking
- **Under the Hood**: Model diagnostics — metrics, feature importance, residual analysis

## How This Maps to CJ's Career

This project sits at the intersection of CJ Fleming's 15+ years in media sales leadership and his Columbia AI certification:

- **Media Partnerships**: The sponsorship economics model reflects real-world deal structures CJ has negotiated — jersey patches, broadcast integrations, naming rights, and multi-channel activation strategies
- **Financial Modeling**: ROI prediction and media valuation are core to the partnership sales cycle — this tool automates the analysis that typically takes days of manual spreadsheet work
- **AI Application**: XGBoost + SHAP demonstrates practical ML deployment beyond academic exercises — the kind of data-driven sales tool that differentiates modern media organizations

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| ML Model | XGBoost |
| Explainability | SHAP |
| Data Processing | pandas, NumPy, scikit-learn |
| Visualization | matplotlib |
| Serialization | joblib |
