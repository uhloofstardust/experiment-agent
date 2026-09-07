# Validation Sets and Model Selection

A held-out validation set (separate from training and test data) is used
to compare candidate models and hyperparameters without touching the final
test set. This avoids "leaking" information about test performance into
model selection decisions.

Only after model selection is finished should the chosen configuration be
evaluated once on the test set, to get an unbiased estimate of real-world
performance.
