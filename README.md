# Permutation Feature Importance — Diabetes Dataset

A model-agnostic explainability technique applied to a Random Forest model
predicting diabetes progression, using scikit-learn's diabetes dataset.

## What it does
1. Trains a Random Forest Regressor on the diabetes dataset
2. Gets baseline RMSE / R² performance
3. Shuffles each feature's values in the test set and re-measures performance
4. Ranks features by how much shuffling degrades performance
5. Plots the results as a horizontal bar chart

## Why Permutation Feature Importance?
Unlike coefficient-based explanations (e.g. linear regression), Permutation
Feature Importance (PFI) works with any model — including "black box" models
like Random Forests or neural networks — by directly measuring how much a
model's performance depends on each feature, without needing access to the
model's internal structure.

## Setup

```bash
python3 -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Run
Open `permutation_feature_importance.ipynb` in VS Code (with the Jupyter
extension) or Jupyter Lab, select the `venv` kernel, and run cells with
Shift+Enter.

## Key finding
`bmi` is the most important feature by far (importance ≈ 13.75), followed
by `s5` (≈ 9.97). Shuffling `s1` and `s2` barely changes performance —
in fact `s1`'s importance is slightly negative, meaning the model doesn't
meaningfully rely on it.
