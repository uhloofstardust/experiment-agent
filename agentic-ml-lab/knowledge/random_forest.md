# Random Forest

Random forests train many decision trees on bootstrapped samples of the
data and average their predictions. This reduces variance compared to a
single decision tree and is fairly robust to noisy features.

`n_estimators` (number of trees) generally improves performance up to a
point of diminishing returns, at higher compute cost. `max_depth` controls
how deep each tree can grow; unlimited depth (`None`) can overfit on small
datasets, while shallow trees may underfit.

Random forests handle unscaled features well and are less sensitive to
feature scaling than linear models.
