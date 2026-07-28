"""
Model evaluation module.

Responsible for:
- Calculating evaluation metrics
- Displaying confusion matrix
- Returning results for comparison
"""

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql import functions as F


def evaluate_model(predictions):
    """
    Evaluate a trained model.

    Parameters
    ----------
    predictions : Spark DataFrame

    Returns
    -------
    dict
        Dictionary containing model metrics.
    """

    accuracy_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy"
    )

    f1_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1"
    )

    precision_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="weightedPrecision"
    )

    recall_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="weightedRecall"
    )

    accuracy = accuracy_evaluator.evaluate(predictions)
    f1 = f1_evaluator.evaluate(predictions)
    precision = precision_evaluator.evaluate(predictions)
    recall = recall_evaluator.evaluate(predictions)

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def get_confusion_matrix(predictions):
    """
    Generate confusion matrix.

    Parameters
    ----------
    predictions : Spark DataFrame

    Returns
    -------
    numpy.ndarray
    """

    prediction_and_labels = (
        predictions
        .select(
            F.col("prediction").cast("double"),
            F.col("label").cast("double")
        )
        .rdd
        .map(tuple)
    )

    metrics = MulticlassMetrics(prediction_and_labels)

    return metrics.confusionMatrix().toArray()


def print_results(model_name, results):
    """
    Print formatted evaluation results.
    """

    print("=" * 50)
    print(f"Model : {model_name}")
    print("=" * 50)

    print(f"Accuracy : {results['accuracy']:.4f}")
    print(f"F1 Score : {results['f1']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall   : {results['recall']:.4f}")

    print("=" * 50)
