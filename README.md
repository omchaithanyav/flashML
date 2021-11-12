# flashML - AutoML tool

![flashml](https://user-images.githubusercontent.com/45726271/141427996-262fdb57-59e5-4969-b973-2b0807e212be.png)


flashML is a AutoML Python library that finds most accurate machine learning models automatically and efficiently.
It frees users from selecting models and hyper-parameters for each model.

## Installation

```bash
pip install flashML
```
## Quickstart

```python
from flashML import autoML
aml = autoML()
aml.fit(X_train, X_test, y_train, y_test, "classification", "f1_score")
```
Task can be either classification or regression and metric can be selected accordingly.

hyper-parameter optimization is done using optuna.

After training, use this function to get the best model:

```python
aml.get_best_model()
```

You can use predict() function for custom predicitions.

```python
aml.predict(X_val)
```