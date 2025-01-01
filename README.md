# Adaboost

This repository contains a **custom implementation** of the **AdaBoost** algorithm using a decision tree stump (`DecisionTreeClassifier` with `max_depth=1`) as the weak learner.

---

## Overview

AdaBoost (Adaptive Boosting) is an ensemble method that combines multiple weak learners to form a strong classifier. The main steps are:

1. **Initialize** sample weights uniformly.
2. **Train** a weak learner (decision stump) on the weighted dataset.
3. **Compute** the weak learner’s weighted error and convert it into an “importance” weight (\(\alpha\)).
4. **Update** the sample weights, giving more focus to previously misclassified samples.
5. **Repeat** steps 2–4 for a specified number of estimators (`n_estimators`).
6. **Aggregate** all weak learners’ predictions using their respective \(\alpha\) values.

## Files

- **`Adaboost.py`**  
  Contains the `AdaBoost` class implementing the AdaBoost algorithm.

## Requirements

- **Python 3.7+** (or higher)  
- **NumPy**  
- **scikit-learn**

You can install the dependencies via:
```bash
pip install numpy scikit-learn
```

## Usage

1. **Import** the custom AdaBoost:
   ```python
   from adaboost_custom import AdaBoost
   ```
2. **Fit** the model on training data:
   ```python
   model = AdaBoost(n_estimators=50, random_state=42)
   model.fit(X_train, y_train)
   ```
3. **Predict** on test data:
   ```python
   y_pred = model.predict(X_test)
   ```
4. **Evaluate** using any scikit-learn metric (e.g., `accuracy_score`):
   ```python
   from sklearn.metrics import accuracy_score
   print("Accuracy:", accuracy_score(y_test, y_pred))
   ```
