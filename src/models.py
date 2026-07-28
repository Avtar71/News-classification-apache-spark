"""
Machine Learning models module.

Responsible for:
- Creating ML models
- Building Spark ML Pipelines
"""

from pyspark.ml import Pipeline

from pyspark.ml.classification import (
    LogisticRegression,
    NaiveBayes,
    RandomForestClassifier,
)

from src.config import (
    FEATURES_COLUMN,
    LABEL_COLUMN,
    MAX_ITER,
    REG_PARAM,
    RANDOM_STATE,
    NUM_TREES,
    MAX_DEPTH,
)

from src.features import get_feature_pipeline


def get_logistic_regression():
    """
    Create Logistic Regression model.
    """

    return LogisticRegression(
        featuresCol=FEATURES_COLUMN,
        labelCol=LABEL_COLUMN,
        maxIter=MAX_ITER,
        regParam=REG_PARAM,
        family="multinomial",
    )


def get_naive_bayes():
    """
    Create Naive Bayes model.
    """

    return NaiveBayes(
        featuresCol=FEATURES_COLUMN,
        labelCol=LABEL_COLUMN,
    )


def get_random_forest():
    """
    Create Random Forest model.
    """

    return RandomForestClassifier(
        featuresCol=FEATURES_COLUMN,
        labelCol=LABEL_COLUMN,
        numTrees=NUM_TREES,
        maxDepth=MAX_DEPTH,
        seed=RANDOM_STATE,
    )


def build_pipeline(model):
    """
    Combine feature engineering with any ML model.

    Parameters
    ----------
    model
        Spark ML model.

    Returns
    -------
    Pipeline
    """

    stages = get_feature_pipeline() + [model]

    return Pipeline(stages=stages)


def get_model(model_name):
    """
    Factory method for selecting ML models.
    """

    models = {
        "logistic_regression": get_logistic_regression(),
        "naive_bayes": get_naive_bayes(),
        "random_forest": get_random_forest(),
    }

    if model_name not in models:
        raise ValueError(f"Unsupported model: {model_name}")

    return build_pipeline(models[model_name])
