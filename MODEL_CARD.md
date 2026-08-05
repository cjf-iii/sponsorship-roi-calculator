# Model Card

## System

Three independent XGBoost regressors estimate media-value ratio, brand-lift percentage, and a composite ROI score from sponsorship-deal attributes.

## Data and split

- 5,000 deterministic synthetic sponsorship deals;
- an 80/20 train/held-out split using seed 42;
- sport, market, deal type, annual spend, term, reach, social following, brand category, and five activation-channel flags;
- no customer, personal, or confidential data.

## Evaluation

The pipeline writes MAE, R-squared, and MAPE for every target to `model/metrics.json`. The current verified run measured ROI R-squared of 0.7771, which rounds to 0.78 in résumé presentation.

## Runtime boundary

XGBoost training is CPU-based. The project records one Apple-silicon verification run and repeats the same locked contract on a public Linux runner; it does not claim Metal acceleration.

## Limitations

Synthetic relationships make the project reproducible and safe to inspect, but the scores do not establish accuracy on real sponsorships. The output is a decision-support demonstration, not a valuation or financial recommendation.

The saved model uses joblib/pickle serialization. Load only artifacts created locally or obtained from a trusted source.
