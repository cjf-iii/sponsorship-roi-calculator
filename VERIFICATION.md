# Verification Record

Verified locally on August 4, 2026. GitHub Actions is configured to repeat the locked contract on a public Linux runner after the updated commit is published.

## Local host

- macOS 26.6;
- Apple M3 Max, 40-core GPU;
- ARM64;
- Metal 4;
- Python 3.12.13.

The models are CPU workloads; the host description is evidence of one successful environment, not a runtime restriction.

## Observed result

All 4 tests and every reviewer-notebook cell passed. The end-to-end pipeline regenerated exactly 5,000 synthetic deals, engineered 117 model inputs, trained three XGBoost regressors, preserved 1,000 untouched observations per target, and passed the fail-closed artifact verifier.

| Target | Held-out R² | MAE | MAPE |
|---|---:|---:|---:|
| Media-value ratio | 0.6802 | 0.2547 | 9.47% |
| Brand-lift percentage | 0.7186 | 0.8189 | 11.90% |
| ROI score | 0.7771 | 5.7473 | 10.63% |

The measured ROI R² of 0.7771 supports the résumé’s conventional two-decimal presentation of 0.78. Every reviewer-notebook cell executed successfully. Streamlit’s application test completed without an exception and rendered diagnostics from the held-out residual arrays.

## Reproduce

```bash
uv sync --locked --python 3.12 --group dev --group notebook
uv run ruff check .
uv run ruff format --check .
uv run python -m unittest discover -s tests -v
uv run --group notebook jupyter nbconvert --to notebook --execute notebooks/01_data_to_validation.ipynb --stdout > /dev/null
uv run python data/generate_data.py
uv run python model/train.py
uv run python scripts/verify_artifacts.py
uv run python -c 'from streamlit.testing.v1 import AppTest; app = AppTest.from_file("app.py", default_timeout=60).run(); assert not app.exception'
```
