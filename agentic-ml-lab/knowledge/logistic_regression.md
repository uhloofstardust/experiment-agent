# Logistic Regression

Logistic regression is a linear model for binary classification. It works
well when classes are roughly linearly separable in feature space, and is
fast, stable, and easy to interpret.

The regularization strength `C` controls overfitting: smaller `C` means
stronger regularization (simpler model, less overfitting, possibly
underfitting); larger `C` means weaker regularization (fits training data
more closely, risk of overfitting).

Because logistic regression is sensitive to feature scale, standardizing
features (zero mean, unit variance) before training usually improves
convergence and performance.
