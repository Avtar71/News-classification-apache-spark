"""
Feature engineering module.

Responsible for:
- Tokenization
- Stopword removal
- TF generation
- TF-IDF feature creation
"""

from pyspark.ml.feature import (
    Tokenizer,
    StopWordsRemover,
    HashingTF,
    IDF,
)

from src.config import (
    CLEAN_TEXT_COLUMN,
    WORDS_COLUMN,
    FILTERED_WORDS_COLUMN,
    RAW_FEATURES_COLUMN,
    FEATURES_COLUMN,
    NUM_FEATURES,
    MIN_DOC_FREQ,
)


def get_feature_pipeline():
    """
    Build the Spark ML feature engineering pipeline.

    Returns
    -------
    list
        List of Spark ML stages.
    """

    tokenizer = Tokenizer(
        inputCol=CLEAN_TEXT_COLUMN,
        outputCol=WORDS_COLUMN,
    )

    stopword_remover = StopWordsRemover(
        inputCol=WORDS_COLUMN,
        outputCol=FILTERED_WORDS_COLUMN,
    )

    hashing_tf = HashingTF(
        inputCol=FILTERED_WORDS_COLUMN,
        outputCol=RAW_FEATURES_COLUMN,
        numFeatures=NUM_FEATURES,
    )

    idf = IDF(
        inputCol=RAW_FEATURES_COLUMN,
        outputCol=FEATURES_COLUMN,
        minDocFreq=MIN_DOC_FREQ,
    )

    return [
        tokenizer,
        stopword_remover,
        hashing_tf,
        idf,
    ]
