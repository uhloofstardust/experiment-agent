# Feature Scaling

Many algorithms (logistic regression, SVMs, k-NN, neural networks) are
sensitive to the scale of input features, since features with larger
numeric ranges can dominate the objective function. Standardization
(subtracting the mean and dividing by the standard deviation) is a common
preprocessing step.

Tree-based models (random forest, gradient boosting) are generally
insensitive to monotonic feature scaling, since they split on thresholds
rather than distances.
