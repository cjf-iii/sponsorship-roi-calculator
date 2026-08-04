# Verification Record

Verified locally on August 4, 2026.

## Host

- macOS 26.6;
- Apple M3 Max, 40-core GPU;
- ARM64;
- Metal 4;
- Python 3.12.13.

## Result

The Metal preflight passed and all 3 unit tests passed. The full pipeline then regenerated 5,000 synthetic deals, engineered 117 model inputs, trained three XGBoost regressors, and evaluated an untouched 20% test partition.

| Target | Held-out R² | MAE | MAPE |
|---|---:|---:|---:|
| Media-value ratio | 0.6802 | 0.2547 | 9.47% |
| Brand-lift percentage | 0.7186 | 0.8189 | 11.90% |
| ROI score | 0.7771 | 5.7473 | 10.63% |

The measured ROI R² of 0.7771 supports the résumé’s conventional two-decimal presentation of 0.78.

## Reproduce

```bash
./scripts/verify_macos_metal.sh
python -m unittest discover -s tests -v
python data/generate_data.py
python model/train.py
```
