import os
import time
import sys
from typing import Tuple

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# Configuration constants
# -----------------------------------------------------------------------------
DATASET_FILENAME = "dataset.csv"  # Expected CSV filename placed in project root or provided via CLI
OUTPUT_DIR = "output"
POSITIVE_THRESHOLD = 4  # Threshold review_score considered positive; adjust if scale differs
TOP_N = 10
SCORE_DISTRIBUTION_MAX_ROWS = 500000  # Cap rows converted to pandas for score distribution to avoid memory issues

# -----------------------------------------------------------------------------
# Utility timing decorator for comparing RDD vs DataFrame performance
# -----------------------------------------------------------------------------

def timed(label: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start
            print(f"[TIMING] {label}: {duration:.4f} seconds")
            return result
        return wrapper
    return decorator

# -----------------------------------------------------------------------------
# Spark initialization
# -----------------------------------------------------------------------------

def init_spark(app_name: str = "SteamReviewsAnalysis") -> SparkSession:
    # Ensure workers use same Python executable (helps on Windows where PATH may differ)
    return (SparkSession.builder
            .appName(app_name)
            .config("spark.pyspark.driver.python", sys.executable)
            .config("spark.pyspark.python", sys.executable)
            .getOrCreate())

# -----------------------------------------------------------------------------
# Data ingestion
# -----------------------------------------------------------------------------
@timed("Load CSV DataFrame")
def load_dataset(spark: SparkSession, path: str) -> DataFrame:
    df = (spark.read.option("header", True)
                  .option("inferSchema", True)
                  .csv(path))
    return df

# -----------------------------------------------------------------------------
# Data cleaning
# -----------------------------------------------------------------------------
@timed("Clean DataFrame")
def clean_reviews(df: DataFrame) -> DataFrame:
    # Remove rows where review_text is null/empty; trim whitespace first
    cleaned = (df.withColumn("review_text", F.trim(F.col("review_text")))
                 .filter(F.col("review_text").isNotNull() & (F.length(F.col("review_text")) > 0)))
    # Ensure review_score is numeric and non-null for aggregations; drop rows where score is null
    if "review_score" in cleaned.columns:
        cleaned = cleaned.filter(F.col("review_score").isNotNull())
    # Cast review_score to integer if inferred differently
    if "review_score" in cleaned.columns and dict(cleaned.dtypes)["review_score"] != "int":
        cleaned = cleaned.withColumn("review_score", F.col("review_score").cast("int"))
    # Remove exact duplicate rows based on identifying columns (game_id if present else game + review_text)
    if "game_id" in cleaned.columns:
        cleaned = cleaned.dropDuplicates(["game_id", "review_text"])
    else:
        cleaned = cleaned.dropDuplicates(["game", "review_text"])  # Fallback
    return cleaned

# -----------------------------------------------------------------------------
# RDD MapReduce style aggregations
# -----------------------------------------------------------------------------
@timed("RDD MapReduce Aggregations")
def rdd_aggregations(df: DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Select only relevant columns and convert to RDD (standardized 'game')
    rdd = df.select("game", "review_score").filter(F.col("review_score").isNotNull()).rdd

    # 1. Count reviews per game
    review_counts_rdd = rdd.map(lambda row: (row[0], 1)).reduceByKey(lambda a, b: a + b)

    # 2. Average review_score per game
    score_pairs_rdd = rdd.map(lambda row: (row[0], (int(row[1]), 1)))
    sum_count_rdd = score_pairs_rdd.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
    avg_score_rdd = sum_count_rdd.mapValues(lambda v: v[0] / v[1] if v[1] else None)

    # 3. Sentiment counts per game (positive / negative based on threshold)
    sentiment_pairs_rdd = rdd.map(lambda row: (row[0], (1, 0) if int(row[1]) >= POSITIVE_THRESHOLD else (0, 1)))
    sentiment_counts_rdd = sentiment_pairs_rdd.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))

    # Convert to pandas DataFrames for easier merging/comparison & visualization
    review_counts_pd = pd.DataFrame(review_counts_rdd.collect(), columns=["game", "review_count_rdd"])
    avg_scores_pd = pd.DataFrame(avg_score_rdd.collect(), columns=["game", "avg_review_score_rdd"])
    sentiment_pd = pd.DataFrame(sentiment_counts_rdd.collect(), columns=["game", "positive_reviews_rdd", "negative_reviews_rdd"])

    return review_counts_pd, avg_scores_pd, sentiment_pd

# -----------------------------------------------------------------------------
# DataFrame API aggregations (for comparison)
# -----------------------------------------------------------------------------
@timed("DataFrame Aggregations")
def dataframe_aggregations(df: DataFrame) -> Tuple[DataFrame, DataFrame, DataFrame]:
    review_counts_df = df.groupBy("game").agg(F.count(F.lit(1)).alias("review_count_df"))
    avg_scores_df = df.groupBy("game").agg(F.avg("review_score").alias("avg_review_score_df"))
    sentiment_df = (df.withColumn("sentiment", F.when(F.col("review_score") >= POSITIVE_THRESHOLD, "positive").otherwise("negative"))
                      .groupBy("game", "sentiment")
                      .agg(F.count(F.lit(1)).alias("sentiment_count"))
                      .groupBy("game")
                      .pivot("sentiment", ["positive", "negative"]).agg(F.first("sentiment_count"))
                      .fillna(0)
                      .withColumnRenamed("positive", "positive_reviews_df")
                      .withColumnRenamed("negative", "negative_reviews_df"))
    return review_counts_df, avg_scores_df, sentiment_df

# -----------------------------------------------------------------------------
# Exploratory Data Analysis helpers
# -----------------------------------------------------------------------------
@timed("EDA Computations")
def eda(df: DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Top N games by review count
    top_games_df = (df.groupBy("game")
                      .agg(F.count(F.lit(1)).alias("review_count"))
                      .orderBy(F.col("review_count").desc())
                      .limit(TOP_N))

    # Distribution of review scores (collect to pandas)
    # Limit rows for distribution to reduce driver memory usage on large datasets
    score_distribution_df = df.select("review_score").limit(SCORE_DISTRIBUTION_MAX_ROWS)

    # Average score per app_name (if present)
    if "app_name" in df.columns:
        app_avg_df = (df.groupBy("app_name")
                        .agg(F.avg("review_score").alias("avg_review_score"))
                        .orderBy(F.col("avg_review_score").desc()))
    else:
        # Fallback: reuse game average
        app_avg_df = (df.groupBy("game")
                        .agg(F.avg("review_score").alias("avg_review_score"))
                        .orderBy(F.col("avg_review_score").desc()))

    return (top_games_df.toPandas(),
            score_distribution_df.toPandas(),
            app_avg_df.toPandas())

# -----------------------------------------------------------------------------
# Visualization functions
# -----------------------------------------------------------------------------
@timed("Generate Visualizations")
def generate_visualizations(top_games_pd: pd.DataFrame,
                            score_distribution_pd: pd.DataFrame,
                            app_avg_pd: pd.DataFrame,
                            output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    sns.set(style="whitegrid")

    # Bar chart: Top N games by review count
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_games_pd, x="review_count", y="game")
    plt.title(f"Top {TOP_N} Games by Review Count")
    plt.xlabel("Review Count")
    plt.ylabel("Game")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_games_review_count.png"))
    plt.close()

    # Histogram: Review score distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(score_distribution_pd["review_score"], bins=20, kde=True, color="steelblue")
    plt.title("Distribution of Review Scores")
    plt.xlabel("Review Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "review_score_distribution.png"))
    plt.close()

    # Bar chart: Average review score for top games (reusing top_games list joined with avg scores)
    # Merge with average scores for clarity
    avg_scores_subset = app_avg_pd.head(TOP_N)
    plt.figure(figsize=(10, 6))
    target_y = avg_scores_subset.columns[0]
    sns.barplot(data=avg_scores_subset, x="avg_review_score", y=target_y)
    plt.title(f"Average Review Score - Top {TOP_N} (By Avg Score)")
    plt.xlabel("Average Review Score")
    plt.ylabel(target_y)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "average_review_score_top.png"))
    plt.close()

# -----------------------------------------------------------------------------
# Save outputs helper
# -----------------------------------------------------------------------------
@timed("Save Output Data")
def save_outputs(output_dir: str,
                 rdd_counts_pd: pd.DataFrame, rdd_avg_pd: pd.DataFrame, rdd_sentiment_pd: pd.DataFrame,
                 df_counts: DataFrame, df_avg: DataFrame, df_sentiment: DataFrame,
                 top_games_pd: pd.DataFrame, score_distribution_pd: pd.DataFrame, app_avg_pd: pd.DataFrame) -> None:
    os.makedirs(output_dir, exist_ok=True)

    rdd_counts_pd.to_csv(os.path.join(output_dir, "rdd_review_counts.csv"), index=False)
    rdd_avg_pd.to_csv(os.path.join(output_dir, "rdd_avg_scores.csv"), index=False)
    rdd_sentiment_pd.to_csv(os.path.join(output_dir, "rdd_sentiment_counts.csv"), index=False)

    df_counts.coalesce(1).write.mode("overwrite").option("header", True).csv(os.path.join(output_dir, "df_review_counts"))
    df_avg.coalesce(1).write.mode("overwrite").option("header", True).csv(os.path.join(output_dir, "df_avg_scores"))
    df_sentiment.coalesce(1).write.mode("overwrite").option("header", True).csv(os.path.join(output_dir, "df_sentiment_counts"))

    top_games_pd.to_csv(os.path.join(output_dir, "top_games.csv"), index=False)
    score_distribution_pd.to_csv(os.path.join(output_dir, "score_distribution.csv"), index=False)
    app_avg_pd.to_csv(os.path.join(output_dir, "app_avg_scores.csv"), index=False)


def main(dataset_path: str = None):
    spark = init_spark()

    if dataset_path is None:
        candidate = os.path.join(os.path.dirname(os.path.dirname(__file__)), DATASET_FILENAME)
        if not os.path.exists(candidate):
            raise FileNotFoundError(f"Dataset not found at {candidate}. Provide path via --path argument.")
        dataset_path = candidate

    print(f"Using dataset path: {dataset_path}")

    raw_df = load_dataset(spark, dataset_path)
    print("Schema:")
    raw_df.printSchema()

    df_std = raw_df
    if "Game Name" in df_std.columns and "game" not in df_std.columns:
        df_std = df_std.withColumnRenamed("Game Name", "game")
    elif "app_name" in df_std.columns and "game" not in df_std.columns:
        df_std = df_std.withColumnRenamed("app_name", "game")
    if "Game ID" in df_std.columns and "game_id" not in df_std.columns:
        df_std = df_std.withColumnRenamed("Game ID", "game_id")
    elif "app_id" in df_std.columns and "game_id" not in df_std.columns:
        df_std = df_std.withColumnRenamed("app_id", "game_id")

    cleaned_df = clean_reviews(df_std)
    print(f"Cleaned count: {cleaned_df.count()} (original {raw_df.count()})")

    try:
        rdd_review_counts_pd, rdd_avg_scores_pd, rdd_sentiment_pd = rdd_aggregations(cleaned_df)
    except Exception as e:
        print(f"[WARN] RDD aggregations failed: {e}. Falling back to DataFrame-only results.")
        rdd_review_counts_pd = pd.DataFrame(columns=["game", "review_count_rdd"])
        rdd_avg_scores_pd = pd.DataFrame(columns=["game", "avg_review_score_rdd"])
        rdd_sentiment_pd = pd.DataFrame(columns=["game", "positive_reviews_rdd", "negative_reviews_rdd"])

    df_review_counts, df_avg_scores, df_sentiment = dataframe_aggregations(cleaned_df)

    top_games_pd, score_distribution_pd, app_avg_pd = eda(cleaned_df)

    generate_visualizations(top_games_pd, score_distribution_pd, app_avg_pd, os.path.join(os.path.dirname(os.path.dirname(__file__)), OUTPUT_DIR))

    save_outputs(os.path.join(os.path.dirname(os.path.dirname(__file__)), OUTPUT_DIR),
                 rdd_review_counts_pd, rdd_avg_scores_pd, rdd_sentiment_pd,
                 df_review_counts, df_avg_scores, df_sentiment,
                 top_games_pd, score_distribution_pd, app_avg_pd)

    print("Processing complete. Outputs saved to 'output' directory.")
    spark.stop()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Steam game reviews PySpark analysis")
    parser.add_argument("--path", type=str, help="Path to dataset.csv", required=False)
    args = parser.parse_args()
    main(dataset_path=args.path)
