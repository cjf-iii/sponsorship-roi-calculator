# Notebook workflow

Launch the executable tutorial from this repository root:

```bash
uv sync --locked --group notebook
uv run --group notebook jupyter lab notebooks/01_data_to_validation.ipynb
```

The notebook generates a disorganized 5,000-deal export and its cleaned counterpart under `notebooks/data/`, trains the real three-model pipeline under `notebooks/artifacts/`, computes SHAP importance, and runs the unit tests.
