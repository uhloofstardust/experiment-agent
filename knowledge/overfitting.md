# Overfitting and Underfitting

Overfitting happens when a model learns noise or idiosyncrasies of the
training set rather than the underlying pattern, leading to a large gap
between training and validation performance. Underfitting happens when a
model is too simple to capture the pattern at all.

Common signs of overfitting: very high training score but noticeably lower
validation score, or a validation score that improves then degrades as
model complexity increases. Regularization, simpler models, more data, and
early stopping all help control overfitting.
