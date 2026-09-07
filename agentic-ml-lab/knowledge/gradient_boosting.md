# Gradient Boosting

Gradient boosting builds trees sequentially, where each new tree corrects
the errors of the previous ensemble. It often achieves strong performance
on tabular data but is more prone to overfitting than random forests if
not tuned carefully.

Key hyperparameters: `learning_rate` (step size — lower values need more
estimators but generalize better), `n_estimators` (number of boosting
rounds), and `max_depth` (depth of each weak learner, usually kept shallow,
e.g. 2-5).

A common strategy is to lower the learning rate while increasing
n_estimators to trade compute for generalization.
