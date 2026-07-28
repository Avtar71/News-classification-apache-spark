"""
Text preprocessing module.

Responsible for:
- Cleaning raw text
- Combining title and description
- Preparing text for feature engineering
"""

import re

from pyspark.sql import functions as F
from pyspark.sql.types import StringType

from src.config import (
    TITLE_COLUMN,
    DESCRIPTION_COLUMN,
    RAW_TEXT_COLUMN,
    CLEAN_TEXT_COLUMN
)


@F.udf(StringType())
def clean_text(text):
    """
    Clean raw news text.

    Steps:
    1. Convert to lowercase
    2. Remove URLs
    3. Remove non-alphabetic characters
    4. Remove extra whitespace
    """

    if text is None:
        return ""

    text = text.lower()

    # Remove URLs
    text = re.sub(r"https?://\S+", " ", text)

    # Remove non-alphabetic characters
    text = re.sub(r"[^a-z\s]", " ", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_dataframe(df):
    """
    Create cleaned text column for ML models.

    Parameters
    ----------
    df : Spark DataFrame

    Returns
    -------
    Spark DataFrame
    """

    return (
        df
        .withColumn(
            RAW_TEXT_COLUMN,
            F.concat_ws(
                " ",
                F.col(TITLE_COLUMN),
                F.col(DESCRIPTION_COLUMN)
            )
        )
        .withColumn(
            CLEAN_TEXT_COLUMN,
            clean_text(F.col(RAW_TEXT_COLUMN))
        )
    )
