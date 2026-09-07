# Hyperparameter Search Strategy

When experiment budget is limited, a reasonable strategy is: (1) establish
a simple baseline, (2) try a couple of structurally different model
families (e.g. linear vs. tree ensemble), (3) take the best-performing
family and spend remaining budget tuning its most impactful
hyperparameters rather than exploring many families shallowly.

Prioritize hyperparameters with the largest expected effect on the target
metric (e.g. regularization strength or tree depth) over minor ones.
