import os
import time
import sys
from typing import Tuple
from wordcloud import WordCloud
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
            .getOrCreate())

@timed("Load CSV DataFrame")
def load_dataset(spark: SparkSession, path: str) -> DataFrame:
    df = (spark.read.option("header", True)
                  .option("inferSchema", True)
                  .csv(path))
    return df

@timed("Clean DataFrame")
def clean_reviews(df: DataFrame) -> DataFrame:
    cleaned = (df.withColumn("review_text", F.trim(F.col("review_text")))
                 .filter(F.col("review_text").isNotNull() & (F.length(F.col("review_text")) > 0)))
    if "review_score" in cleaned.columns:
        cleaned = cleaned.filter(F.col("review_score").isNotNull())
    if "review_score" in cleaned.columns and dict(cleaned.dtypes)["review_score"] != "int":
        cleaned = cleaned.withColumn("review_score", F.col("review_score").cast("int"))
    if "game_id" in cleaned.columns:
        cleaned = cleaned.dropDuplicates(["game_id", "review_text"])
    else:
        cleaned = cleaned.dropDuplicates(["game", "review_text"])
    return cleaned

@timed("RDD MapReduce Aggregations")
def rdd_aggregations(df: DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rdd = df.select("game", "review_score").filter(F.col("review_score").isNotNull()).rdd
    review_counts_rdd = rdd.map(lambda row: (row[0], 1)).reduceByKey(lambda a, b: a + b)
    score_pairs_rdd = rdd.map(lambda row: (row[0], (int(row[1]), 1)))
    sum_count_rdd = score_pairs_rdd.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
    avg_score_rdd = sum_count_rdd.mapValues(lambda v: v[0] / v[1] if v[1] else None)
    sentiment_pairs_rdd = rdd.map(lambda row: (row[0], (1, 0) if int(row[1]) >= POSITIVE_THRESHOLD else (0, 1)))
    sentiment_counts_rdd = sentiment_pairs_rdd.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
    review_counts_pd = pd.DataFrame(review_counts_rdd.collect(), columns=["game", "review_count_rdd"])
    avg_scores_pd = pd.DataFrame(avg_score_rdd.collect(), columns=["game", "avg_review_score_rdd"])
    sentiment_pd = pd.DataFrame(sentiment_counts_rdd.collect(), columns=["game", "positive_reviews_rdd", "negative_reviews_rdd"])

    return review_counts_pd, avg_scores_pd, sentiment_pd

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

@timed("Extended EDA (length, word frequencies, timestamps)")
def extended_eda(df: DataFrame, output_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    os.makedirs(output_dir, exist_ok=True)

    df_len = df.withColumn("review_length", F.length(F.col("review_text")))
    length_pd = df_len.select("review_length").toPandas()
    length_pd.to_csv(os.path.join(output_dir, "review_lengths.csv"), index=False)

    df_words = df.withColumn("word", F.explode(F.split(F.col("review_text"), "\s+")))
    word_counts_df = (df_words.groupBy("word")
                      .agg(F.count("*").alias("count"))
                      .orderBy(F.col("count").desc()))
    word_counts_pd = word_counts_df.toPandas()
    word_counts_pd.to_csv(os.path.join(output_dir, "word_counts.csv"), index=False)

    if "timestamp_created" in df.columns:
        df_time = (df
                   .withColumn("year", F.year("timestamp_created"))
                   .withColumn("month", F.month("timestamp_created")))
    else:
        df_time = df.withColumn("year", F.lit(None)).withColumn("month", F.lit(None))

    year_pd = (df_time.groupBy("year")
               .agg(F.count("*").alias("review_count"))
               .orderBy("year")
               .toPandas())
    year_pd.to_csv(os.path.join(output_dir, "reviews_per_year.csv"), index=False)

    month_pd = (df_time.groupBy("year", "month")
                .agg(F.count("*").alias("review_count"))
                .orderBy("year", "month")
                .toPandas())
    month_pd.to_csv(os.path.join(output_dir, "reviews_per_month.csv"), index=False)
    text_pd = df.select("review_text").toPandas()
    all_text = " ".join(text_pd["review_text"].astype(str).tolist())

    with open(os.path.join(output_dir, "all_reviews_text.txt"), "w", encoding="utf-8") as f:
        f.write(all_text)

    return length_pd, word_counts_pd, year_pd, month_pd, all_text


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

@timed("Generate Extended Visualizations")
def generate_extended_visualizations(length_pd, word_counts_pd, year_pd, month_pd, all_text, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    sns.set(style="whitegrid")

    plt.figure(figsize=(10, 5))
    sns.histplot(length_pd["review_length"], bins=50, kde=True)
    plt.title("Histogram of Review Lengths")
    plt.xlabel("Review length (characters)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "review_length_histogram.png"))
    plt.close()

    if "year" in year_pd.columns:
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=year_pd, x="year", y="review_count", marker="o")
        plt.title("Number of Reviews per Year")
        plt.xlabel("Year")
        plt.ylabel("Number of Reviews")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "reviews_per_year.png"))
        plt.close()

    if {"year", "month"}.issubset(month_pd.columns):
        plt.figure(figsize=(12, 5))
        sns.lineplot(data=month_pd, x="month", y="review_count", hue="year")
        plt.title("Reviews per Month (Grouped by Year)")
        plt.xlabel("Month")
        plt.ylabel("Review Count")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "reviews_per_month.png"))
        plt.close()

    try:
        wc = WordCloud(width=1600, height=800, background_color="white").generate(all_text)
        plt.figure(figsize=(14, 7))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "wordcloud.png"))
        plt.close()
        print("[INFO] WordCloud generated successfully.")
    except Exception as e:
        print(f"[WARN] WordCloud generation failed: {e}")

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
    length_pd, word_counts_pd, year_pd, month_pd, all_text = extended_eda(
        cleaned_df,
        os.path.join(os.path.dirname(os.path.dirname(__file__)), OUTPUT_DIR)
    )

    generate_extended_visualizations(
        length_pd, word_counts_pd, year_pd, month_pd, all_text,
        os.path.join(os.path.dirname(os.path.dirname(__file__)), OUTPUT_DIR)
    )
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
