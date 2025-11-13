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

DATASET_FILENAME = "dataset.csv"
OUTPUT_DIR = "output"
POSITIVE_THRESHOLD = 4  
TOP_N = 10
SCORE_DISTRIBUTION_MAX_ROWS = 500000

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

def init_spark(app_name: str = "SteamReviewsAnalysis") -> SparkSession:
    return (SparkSession.builder
            .appName(app_name)
            .config("spark.pyspark.driver.python", sys.executable)
            .config("spark.pyspark.python", sys.executable)
            .config("spark.driver.memory", "6g")
            .config("spark.executor.memory", "6g")
            .config("spark.sql.shuffle.partitions", "48")
            .config("spark.memory.fraction", "0.6")
            .getOrCreate())

def detect_score_scale(df: DataFrame):
    stats = df.select(F.min("review_score").alias("min"),
                      F.max("review_score").alias("max"),
                      F.countDistinct("review_score").alias("distinct"))
    row = stats.collect()[0]
    mn, mx, distinct = row[0], row[1], row[2]
    is_binary = distinct <= 3 and mn is not None and mx is not None and mn >= -1 and mx <= 1
    if is_binary:
        threshold = 0
    elif mx is not None and mx <= 5:
        threshold = 4
    elif mx is not None and mx <= 10:
        threshold = 7
    else:
        quant = df.approxQuantile("review_score", [0.5], 0.01)
        threshold = quant[0] if quant else POSITIVE_THRESHOLD
    return {"min": mn, "max": mx, "distinct": distinct, "binary": is_binary, "threshold": threshold}

@timed("Load CSV DataFrame")
def load_dataset(spark: SparkSession, path: str) -> DataFrame:
    df = (spark.read.option("header", True)
                  .option("inferSchema", True)
                  .csv(path))
    return df

@timed("Clean DataFrame")
def clean_reviews(df: DataFrame, skip_dedup: bool = False) -> DataFrame:
    cleaned = (df.withColumn("review_text", F.trim(F.col("review_text")))
                 .filter(F.col("review_text").isNotNull() & (F.length(F.col("review_text")) > 0)))
    if "review_score" in cleaned.columns:
        cleaned = cleaned.filter(F.col("review_score").isNotNull())
    # Standardize numeric type (prefer double precision) without truncation
    if "review_score" in cleaned.columns and dict(cleaned.dtypes)["review_score"] not in ("double", "float"):
        cleaned = cleaned.withColumn("review_score", F.col("review_score").cast("double"))
    if not skip_dedup:
        # Deduplication can be memory heavy; allow disabling via flag
        if "game_id" in cleaned.columns:
            cleaned = cleaned.dropDuplicates(["game_id", "review_text"])
        else:
            cleaned = cleaned.dropDuplicates(["game", "review_text"])
    return cleaned

@timed("RDD MapReduce Aggregations")
def rdd_aggregations(df: DataFrame, threshold: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rdd = df.select("game", "review_score").filter(F.col("review_score").isNotNull()).rdd
    review_counts_rdd = rdd.map(lambda row: (row[0], 1)).reduceByKey(lambda a, b: a + b)
    score_pairs_rdd = rdd.map(lambda row: (row[0], (float(row[1]), 1)))
    sum_count_rdd = score_pairs_rdd.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
    avg_score_rdd = sum_count_rdd.mapValues(lambda v: v[0] / v[1] if v[1] else None)
    sentiment_pairs_rdd = rdd.map(lambda row: (row[0], (1, 0) if float(row[1]) >= threshold else (0, 1)))
    sentiment_counts_rdd = sentiment_pairs_rdd.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
    review_counts_pd = pd.DataFrame(review_counts_rdd.collect(), columns=["game", "review_count_rdd"])
    avg_scores_pd = pd.DataFrame(avg_score_rdd.collect(), columns=["game", "avg_review_score_rdd"])
    sentiment_pd = pd.DataFrame(sentiment_counts_rdd.collect(), columns=["game", "positive_reviews_rdd", "negative_reviews_rdd"])
    if not sentiment_pd.empty:
        sentiment_pd["total_reviews_rdd"] = sentiment_pd["positive_reviews_rdd"] + sentiment_pd["negative_reviews_rdd"]
        sentiment_pd["positive_ratio_rdd"] = sentiment_pd.apply(lambda r: (r["positive_reviews_rdd"] / r["total_reviews_rdd"]) if r["total_reviews_rdd"] else None, axis=1)
    return review_counts_pd, avg_scores_pd, sentiment_pd

@timed("DataFrame Aggregations")
def dataframe_aggregations(df: DataFrame, threshold: float) -> Tuple[DataFrame, DataFrame, DataFrame]:
    review_counts_df = df.groupBy("game").agg(F.count(F.lit(1)).alias("review_count_df"))
    avg_scores_df = df.groupBy("game").agg(F.avg("review_score").alias("avg_review_score_df"))
    sentiment_df = (df.withColumn("sentiment", F.when(F.col("review_score") >= F.lit(threshold), "positive").otherwise("negative"))
                      .groupBy("game", "sentiment")
                      .agg(F.count(F.lit(1)).alias("sentiment_count"))
                      .groupBy("game")
                      .pivot("sentiment", ["positive", "negative"]).agg(F.first("sentiment_count"))
                      .fillna(0)
                      .withColumnRenamed("positive", "positive_reviews_df")
                      .withColumnRenamed("negative", "negative_reviews_df"))
    sentiment_df = sentiment_df.withColumn("positive_ratio_df", F.when((F.col("positive_reviews_df") + F.col("negative_reviews_df")) > 0,
                                                                        F.col("positive_reviews_df") / (F.col("positive_reviews_df") + F.col("negative_reviews_df")))
                                                   .otherwise(F.lit(None)))
    return review_counts_df, avg_scores_df, sentiment_df

@timed("EDA Computations")
def eda(df: DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    top_games_df = (df.groupBy("game")
                      .agg(F.count(F.lit(1)).alias("review_count"))
                      .orderBy(F.col("review_count").desc())
                      .limit(TOP_N))
    score_distribution_df = df.select("review_score").limit(SCORE_DISTRIBUTION_MAX_ROWS)
    if "app_name" in df.columns:
        app_avg_df = (df.groupBy("app_name")
                        .agg(F.avg("review_score").alias("avg_review_score"))
                        .orderBy(F.col("avg_review_score").desc()))
    else:
        app_avg_df = (df.groupBy("game")
                        .agg(F.avg("review_score").alias("avg_review_score"))
                        .orderBy(F.col("avg_review_score").desc()))

    return (top_games_df.toPandas(),
            score_distribution_df.toPandas(),
            app_avg_df.toPandas())

@timed("Generate Visualizations")
def generate_visualizations(top_games_pd: pd.DataFrame,
                            score_distribution_pd: pd.DataFrame,
                            app_avg_pd: pd.DataFrame,
                            output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    sns.set(style="whitegrid")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_games_pd, x="review_count", y="game")
    plt.title(f"Top {TOP_N} Games by Review Count")
    plt.xlabel("Review Count")
    plt.ylabel("Game")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_games_review_count.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(score_distribution_pd["review_score"], bins=20, kde=True, color="steelblue")
    plt.title("Distribution of Review Scores")
    plt.xlabel("Review Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "review_score_distribution.png"))
    plt.close()

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
def main(dataset_path: str = None, sample_fraction: float = 1.0, no_dedup: bool = False):
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

    cleaned_df = clean_reviews(df_std, skip_dedup=no_dedup)
    try:
        original_cnt = raw_df.count()
    except Exception:
        original_cnt = None
    try:
        cleaned_cnt = cleaned_df.count()
    except Exception:
        cleaned_cnt = None
    print(f"Counts -> original={original_cnt if original_cnt is not None else 'NA'} cleaned={cleaned_cnt if cleaned_cnt is not None else 'NA'} dedup={'off' if no_dedup else 'on'}")

    if 0 < sample_fraction < 1.0:
        cleaned_df = cleaned_df.sample(False, sample_fraction, seed=42)
        print(f"Sampling applied: fraction={sample_fraction}")

    scale_info = detect_score_scale(cleaned_df)
    threshold = scale_info['threshold']
    print(f"[INFO] score_scale min={scale_info['min']} max={scale_info['max']} distinct={scale_info['distinct']} binary={scale_info['binary']} threshold={threshold}")

    try:
        rdd_review_counts_pd, rdd_avg_scores_pd, rdd_sentiment_pd = rdd_aggregations(cleaned_df, threshold)
    except Exception as e:
        print(f"[WARN] RDD aggregations failed: {e}. Falling back to DataFrame-only results.")
        rdd_review_counts_pd = pd.DataFrame(columns=["game", "review_count_rdd"])
        rdd_avg_scores_pd = pd.DataFrame(columns=["game", "avg_review_score_rdd"])
        rdd_sentiment_pd = pd.DataFrame(columns=["game", "positive_reviews_rdd", "negative_reviews_rdd"])
    df_review_counts, df_avg_scores, df_sentiment = dataframe_aggregations(cleaned_df, threshold)

    top_games_pd, score_distribution_pd, app_avg_pd = eda(cleaned_df)

    generate_visualizations(top_games_pd, score_distribution_pd, app_avg_pd, os.path.join(os.path.dirname(os.path.dirname(__file__)), OUTPUT_DIR))

    save_outputs(os.path.join(os.path.dirname(os.path.dirname(__file__)), OUTPUT_DIR),
                 rdd_review_counts_pd, rdd_avg_scores_pd, rdd_sentiment_pd,
                 df_review_counts, df_avg_scores, df_sentiment,
                 top_games_pd, score_distribution_pd, app_avg_pd)

    scale_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), OUTPUT_DIR, "score_scale_summary.txt")
    with open(scale_path, "w", encoding="utf-8") as fh:
        for k,v in scale_info.items():
            fh.write(f"{k}={v}\n")
    if scale_info.get("binary"):
        global_pos_neg = (cleaned_df.withColumn("is_positive", F.when(F.col("review_score") >= F.lit(threshold), 1).otherwise(0))
                                        .agg(F.sum("is_positive").alias("positive")))
        positive_total = global_pos_neg.collect()[0]["positive"]
        total_reviews = cleaned_df.count()
        negative_total = total_reviews - positive_total
        ratio = (positive_total / total_reviews) if total_reviews else 0
        global_summary_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), OUTPUT_DIR, "global_binary_summary.txt")
        with open(global_summary_path, "w", encoding="utf-8") as gfh:
            gfh.write(f"positive_count={positive_total}\n")
            gfh.write(f"negative_count={negative_total}\n")
            gfh.write(f"positive_ratio={ratio:.6f}\n")
        print(f"[INFO] Binary score global summary written: positive_ratio={ratio:.4f}")
    print("Processing complete. Outputs saved to 'output' directory.")
    spark.stop()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Steam game reviews PySpark analysis")
    parser.add_argument("--path", type=str, help="Path to dataset.csv", required=False)
    parser.add_argument("--sample-fraction", type=float, default=1.0, help="Fraction of data to process (0< f <=1).")
    parser.add_argument("--no-dedup", action="store_true", help="Disable heavy text-based deduplication to reduce memory usage.")
    args = parser.parse_args()
    main(dataset_path=args.path, sample_fraction=args.sample_fraction, no_dedup=args.no_dedup)
