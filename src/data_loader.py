"""
Data loading module.

Responsible for:
- Creating the Spark session
- Loading CSV datasets
- Cleaning column names
- Converting labels
- Returning Spark DataFrames
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.config import (
    APP_NAME,
    MASTER,
    DRIVER_MEMORY,
    SHUFFLE_PARTITIONS,
    LOG_LEVEL,
    TRAIN_PATH,
    TEST_PATH
)


def create_spark_session():
    """
    Create and configure a Spark Session.
    """

    spark = (
        SparkSession.builder
        .appName(APP_NAME)
        .master(MASTER)
        .config("spark.driver.memory", DRIVER_MEMORY)
        .config("spark.sql.shuffle.partitions", SHUFFLE_PARTITIONS)
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel(LOG_LEVEL)

    return spark


def load_dataset(spark, path):
    """
    Load a CSV dataset and prepare it for modeling.

    Parameters
    ----------
    spark : SparkSession
        Active Spark session.

    path : str or Path
        Path to CSV dataset.

    Returns
    -------
    Spark DataFrame
    """

    df = (
        spark.read
        .option("header", True)
        .option("multiLine", True)
        .option("escape", '"')
        .csv(str(path))
        .withColumnRenamed("Class Index", "label_str")
        .withColumnRenamed("Title", "title")
        .withColumnRenamed("Description", "description")
        .dropna(subset=["label_str", "title", "description"])
        .withColumn(
            "label",
            F.col("label_str").cast("integer") - 1
        )
    )

    return df


def load_data():
    """
    Load train and test datasets.

    Returns
    -------
    train_df : Spark DataFrame

    test_df : Spark DataFrame

    spark : SparkSession
    """

    spark = create_spark_session()

    train_df = load_dataset(spark, TRAIN_PATH)
    test_df = load_dataset(spark, TEST_PATH)

    train_df = train_df.repartition(2).cache()
    test_df = test_df.repartition(2).cache()

    return train_df, test_df, spark
