# Xgboost-Feature-Elimination

In this post you will discover how you can estimate the importance of features for a predictive modeling problem using the XGBoost library in Python.

After reading this post you will know:

    How feature importance is calculated using the gradient boosting algorithm.
    How to plot feature importance in Python calculated by the XGBoost model.
    How to use feature importance calculated by XGBoost to perform feature selection.

Generally, importance provides a score that indicates how useful or valuable each feature was in the construction of the boosted decision trees within the model. The more an attribute is used to make key decisions with decision trees, the higher its relative importance.

This importance is calculated explicitly for each attribute in the dataset, allowing attributes to be ranked and compared to each other.

Importance is calculated for a single decision tree by the amount that each attribute split point improves the performance measure, weighted by the number of observations the node is responsible for. The performance measure may be the purity (Gini index) used to select the split points or another more specific error function.

The feature importances are then averaged across all of the the decision trees within the model.
