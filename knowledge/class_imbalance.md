# Class Imbalance and F1 Score

When classes are imbalanced, accuracy can be misleading — a model that
always predicts the majority class can still score high accuracy. F1
score, the harmonic mean of precision and recall, is a better metric in
this setting because it penalizes models that ignore the minority class.

The breast cancer dataset is moderately imbalanced (more benign than
malignant cases). Optimizing for F1 rather than accuracy encourages models
that balance false positives and false negatives rather than just
predicting the majority class.
