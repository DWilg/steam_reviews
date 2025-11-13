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
import numpy as np

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
            .config("spark.driver.memory", "8g")
            .config("spark.driver.maxResultSize", "2g")
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

@timed("Extended EDA (length, word frequencies, timestamps)")
def extended_eda(df: DataFrame, output_dir: str, threshold: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    os.makedirs(output_dir, exist_ok=True)

    # Add sentiment column and review length
    df_len = df.withColumn("review_length", F.length(F.col("review_text")))
    df_sentiment = df_len.withColumn("sentiment", F.when(F.col("review_score") >= F.lit(threshold), "positive").otherwise("negative"))
    
    # Length analysis
    length_pd = df_len.select("review_length").toPandas()
    length_pd.to_csv(os.path.join(output_dir, "review_lengths.csv"), index=False)
    
    # Sentiment vs length analysis
    sentiment_length_pd = df_sentiment.select("sentiment", "review_length").toPandas()
    sentiment_length_pd.to_csv(os.path.join(output_dir, "sentiment_length.csv"), index=False)
    
    # Games sentiment analysis
    games_sentiment_df = (df_sentiment.groupBy("game", "sentiment")
                         .agg(F.count("*").alias("count"))
                         .groupBy("game")
                         .pivot("sentiment", ["positive", "negative"])
                         .agg(F.first("count"))
                         .fillna(0)
                         .withColumn("total_reviews", F.col("positive") + F.col("negative"))
                         .withColumn("positive_ratio", F.col("positive") / F.col("total_reviews"))
                         .filter(F.col("total_reviews") >= 10))  # Filter games with at least 10 reviews
    
    games_sentiment_pd = games_sentiment_df.toPandas()
    games_sentiment_pd.to_csv(os.path.join(output_dir, "games_sentiment.csv"), index=False)
    
    # Correlation data - handle missing columns gracefully
    correlation_cols = []
    if "review_length" in df_sentiment.columns:
        correlation_cols.append("review_length")
    if "review_score" in df_sentiment.columns:
        correlation_cols.append("review_score")
    if "review_votes" in df_sentiment.columns:
        correlation_cols.append("review_votes")
    
    if correlation_cols:
        correlation_df = df_sentiment.select(*correlation_cols).toPandas()
        correlation_df.to_csv(os.path.join(output_dir, "correlation_data.csv"), index=False)
    else:
        # Create empty dataframe if no numeric columns
        correlation_df = pd.DataFrame()

    df_words = df.withColumn("word", F.explode(F.split(F.col("review_text"), r"\s+")))
    word_counts_df = (df_words.groupBy("word")
                      .agg(F.count("*").alias("count"))
                      .orderBy(F.col("count").desc()))
    word_counts_pd = word_counts_df.toPandas()
    word_counts_pd.to_csv(os.path.join(output_dir, "word_counts.csv"), index=False)

    # Check if timestamp column exists (different possible names)
    timestamp_col = None
    for col_name in ["timestamp_created", "timestamp", "date", "created_at"]:
        if col_name in df_sentiment.columns:
            timestamp_col = col_name
            break
    
    if timestamp_col:
        df_time = (df_sentiment
                   .withColumn("year", F.year(timestamp_col))
                   .withColumn("month", F.month(timestamp_col)))
    else:
        # Create dummy data for demonstration if no timestamp
        df_time = df_sentiment.withColumn("year", F.lit(None)).withColumn("month", F.lit(None))

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
    # Sample data for text analysis
    df_sampled_for_text = df_sentiment.sample(False, 0.1, seed=42)
    text_pd = df_sampled_for_text.select("review_text", "sentiment").toPandas()
    
    # Separate positive and negative text
    positive_text = " ".join(text_pd[text_pd["sentiment"] == "positive"]["review_text"].astype(str).tolist())
    negative_text = " ".join(text_pd[text_pd["sentiment"] == "negative"]["review_text"].astype(str).tolist())
    all_text = " ".join(text_pd["review_text"].astype(str).tolist())

    # Save text files
    with open(os.path.join(output_dir, "all_reviews_text.txt"), "w", encoding="utf-8") as f:
        f.write(all_text)
    with open(os.path.join(output_dir, "positive_reviews_text.txt"), "w", encoding="utf-8") as f:
        f.write(positive_text)
    with open(os.path.join(output_dir, "negative_reviews_text.txt"), "w", encoding="utf-8") as f:
        f.write(negative_text)

    return length_pd, word_counts_pd, year_pd, month_pd, sentiment_length_pd, games_sentiment_pd, correlation_df


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
def generate_extended_visualizations(length_pd, word_counts_pd, year_pd, month_pd, sentiment_length_pd, games_sentiment_pd, correlation_df, output_dir):
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

    # Sentiment distribution pie chart
    if "sentiment" in sentiment_length_pd.columns:
        sentiment_counts = sentiment_length_pd["sentiment"].value_counts()
        plt.figure(figsize=(8, 8))
        colors = ['#2ecc71', '#e74c3c']  # Green for positive, red for negative
        plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        plt.title("Overall Sentiment Distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "sentiment_pie_chart.png"))
        plt.close()
    
    # Review length comparison by sentiment
    if "sentiment" in sentiment_length_pd.columns and "review_length" in sentiment_length_pd.columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=sentiment_length_pd, x="sentiment", y="review_length")
        plt.title("Review Length Distribution by Sentiment")
        plt.xlabel("Sentiment")
        plt.ylabel("Review Length (characters)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "sentiment_length_boxplot.png"))
        plt.close()
        
        # Histogram comparison
        plt.figure(figsize=(12, 6))
        positive_lengths = sentiment_length_pd[sentiment_length_pd["sentiment"] == "positive"]["review_length"]
        negative_lengths = sentiment_length_pd[sentiment_length_pd["sentiment"] == "negative"]["review_length"]
        
        plt.hist(positive_lengths, bins=50, alpha=0.7, label="Positive", color='#2ecc71')
        plt.hist(negative_lengths, bins=50, alpha=0.7, label="Negative", color='#e74c3c')
        plt.xlabel("Review Length (characters)")
        plt.ylabel("Frequency")
        plt.title("Review Length Distribution by Sentiment")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "sentiment_length_histogram.png"))
        plt.close()
    
    # Generate word clouds
    try:
        # Read text files for word clouds
        with open(os.path.join(output_dir, "all_reviews_text.txt"), "r", encoding="utf-8") as f:
            all_text = f.read()
        with open(os.path.join(output_dir, "positive_reviews_text.txt"), "r", encoding="utf-8") as f:
            positive_text = f.read()
            if positive_text:
                wc_pos = WordCloud(width=800, height=400, background_color="white", colormap="Greens").generate(positive_text)
                plt.figure(figsize=(12, 6))
                plt.imshow(wc_pos, interpolation="bilinear")
                plt.axis("off")
                plt.title("Positive Reviews Word Cloud")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "wordcloud_positive.png"))
                plt.close()
        
        with open(os.path.join(output_dir, "negative_reviews_text.txt"), "r", encoding="utf-8") as f:
            negative_text = f.read()
            if negative_text:
                wc_neg = WordCloud(width=800, height=400, background_color="white", colormap="Reds").generate(negative_text)
                plt.figure(figsize=(12, 6))
                plt.imshow(wc_neg, interpolation="bilinear")
                plt.axis("off")
                plt.title("Negative Reviews Word Cloud")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "wordcloud_negative.png"))
                plt.close()
        
        # General word cloud
        wc = WordCloud(width=1600, height=800, background_color="white").generate(all_text)
        plt.figure(figsize=(14, 7))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title("Overall Word Cloud")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "wordcloud.png"))
        plt.close()
        print("[INFO] WordClouds generated successfully.")
    except Exception as e:
        print(f"[WARN] WordCloud generation failed: {e}")
    
    # Correlation heatmap
    if not correlation_df.empty:
        plt.figure(figsize=(10, 8))
        corr_matrix = correlation_df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f')
        plt.title("Correlation Matrix: Review Features")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
        plt.close()
    
    # Top games by sentiment
    if not games_sentiment_pd.empty:
        # Top 10 most positive games
        top_positive = games_sentiment_pd.nlargest(10, 'positive_ratio')
        plt.figure(figsize=(12, 8))
        sns.barplot(data=top_positive, y='game', x='positive_ratio', palette='Greens_r')
        plt.title('Top 10 Games with Highest Positive Sentiment')
        plt.xlabel('Positive Review Ratio')
        plt.ylabel('Game')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "top_positive_games.png"))
        plt.close()
        
        # Top 10 most negative games
        bottom_positive = games_sentiment_pd.nsmallest(10, 'positive_ratio')
        plt.figure(figsize=(12, 8))
        sns.barplot(data=bottom_positive, y='game', x='positive_ratio', palette='Reds')
        plt.title('Top 10 Games with Lowest Positive Sentiment')
        plt.xlabel('Positive Review Ratio')
        plt.ylabel('Game')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "bottom_positive_games.png"))
        plt.close()
        
        # Scatter plot: Total reviews vs Positive ratio
        plt.figure(figsize=(12, 8))
        plt.scatter(games_sentiment_pd['total_reviews'], games_sentiment_pd['positive_ratio'], 
                   alpha=0.6, s=50)
        plt.xlabel('Total Number of Reviews')
        plt.ylabel('Positive Review Ratio')
        plt.title('Relationship between Game Popularity and Sentiment')
        plt.xscale('log')  # Log scale for better visualization
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "popularity_vs_sentiment.png"))
        plt.close()
    
    # Violin plot for review lengths by sentiment
    if "sentiment" in sentiment_length_pd.columns and "review_length" in sentiment_length_pd.columns:
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=sentiment_length_pd, x="sentiment", y="review_length")
        plt.title("Review Length Distribution by Sentiment (Violin Plot)")
        plt.xlabel("Sentiment")
        plt.ylabel("Review Length (characters)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "sentiment_length_violin.png"))
        plt.close()
    
    # Time series analysis (if timestamp data exists)
    try:
        if "year" in year_pd.columns and not year_pd['year'].isna().all():
            year_data = year_pd[year_pd['year'].notna()]
            if not year_data.empty and len(year_data) > 1:
                plt.figure(figsize=(12, 6))
                plt.plot(year_data['year'], year_data['review_count'], marker='o')
                plt.title('Review Volume Over Time')
                plt.xlabel('Year')
                plt.ylabel('Number of Reviews')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "reviews_timeline.png"))
                plt.close()
            else:
                print("[INFO] No valid timestamp data available for timeline analysis")
        else:
            print("[INFO] No timestamp data available - skipping timeline analysis")
    except Exception as e:
        print(f"[WARN] Timeline analysis failed: {e}")
    
    # Top words comparison (positive vs negative)
    try:
        # Read word counts and create comparison
        if not word_counts_pd.empty:
            # Get top 15 words overall
            top_words = word_counts_pd.head(15)
            
            plt.figure(figsize=(12, 8))
            sns.barplot(data=top_words, x='count', y='word', palette='viridis')
            plt.title('Top 15 Most Frequent Words in Reviews')
            plt.xlabel('Frequency')
            plt.ylabel('Words')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "top_words_overall.png"))
            plt.close()
            
            # Create a simple word frequency comparison visualization
            plt.figure(figsize=(14, 8))
            plt.barh(range(len(top_words)), top_words['count'], color='skyblue')
            plt.yticks(range(len(top_words)), top_words['word'])
            plt.xlabel('Word Frequency')
            plt.title('Word Frequency Analysis')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "word_frequency_bar.png"))
            plt.close()
    except Exception as e:
        print(f"[WARN] Word frequency analysis failed: {e}")

    # === NOWE WYKRESY ===
    
    # 1. Review Length Distribution
    if "review_length" in sentiment_length_pd.columns:
        plt.figure(figsize=(12, 6))
        plt.hist(sentiment_length_pd['review_length'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(sentiment_length_pd['review_length'].mean(), color='red', linestyle='--', 
                   label=f'Średnia: {sentiment_length_pd["review_length"].mean():.0f}')
        plt.axvline(sentiment_length_pd['review_length'].median(), color='orange', linestyle='--', 
                   label=f'Mediana: {sentiment_length_pd["review_length"].median():.0f}')
        plt.title('Rozkład długości recenzji')
        plt.xlabel('Długość recenzji (znaki)')
        plt.ylabel('Częstotliwość')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "review_length_distribution.png"))
        plt.close()
        
    # 2. Top 20 Games by Review Count
    if not games_sentiment_pd.empty and len(games_sentiment_pd) >= 10:
        top_20_games = games_sentiment_pd.nlargest(20, 'total_reviews')
        plt.figure(figsize=(15, 10))
        sns.barplot(data=top_20_games, y='game', x='total_reviews', palette='viridis')
        plt.title('Top 20 gier według liczby recenzji')
        plt.xlabel('Łączna liczba recenzji')
        plt.ylabel('Gra')
        for i, v in enumerate(top_20_games['total_reviews']):
            plt.text(v + max(top_20_games['total_reviews']) * 0.01, i, str(v), va='center')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "top_20_games_by_reviews.png"))
        plt.close()
        
    # 3. Box Plot of Review Lengths by Sentiment
    if {"review_length", "sentiment"}.issubset(sentiment_length_pd.columns):
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=sentiment_length_pd, x='sentiment', y='review_length', 
                   palette=['lightcoral', 'lightblue'])
        plt.title('Rozkład długości recenzji według sentymentu')
        plt.xlabel('Sentyment')
        plt.ylabel('Długość recenzji (znaki)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "length_by_sentiment_boxplot.png"))
        plt.close()
        
    # 4. Pie Chart of Overall Sentiment Distribution
    if "sentiment" in sentiment_length_pd.columns:
        sentiment_counts = sentiment_length_pd['sentiment'].value_counts()
        plt.figure(figsize=(8, 8))
        colors = ['lightcoral' if x == 'negative' else 'lightblue' for x in sentiment_counts.index]
        plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', 
               colors=colors, startangle=90, textprops={'fontsize': 14})
        plt.title('Ogólny rozkład sentymentu', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "sentiment_pie_chart.png"))
        plt.close()
        
    # 5. Games Popularity vs Quality Scatter (Bubble Chart)
    if not games_sentiment_pd.empty and 'positive_ratio' in games_sentiment_pd.columns:
        plt.figure(figsize=(14, 10))
        scatter = plt.scatter(games_sentiment_pd['positive_ratio'], games_sentiment_pd['total_reviews'],
                            s=games_sentiment_pd['total_reviews']/50, alpha=0.6, 
                            c=games_sentiment_pd['positive_ratio'], cmap='viridis')
        plt.colorbar(scatter, label='Współczynnik pozytywnych')
        plt.title('Popularność vs Jakość gier (rozmiar bąbelka = łączne recenzje)')
        plt.xlabel('Współczynnik pozytywnych recenzji')
        plt.ylabel('Łączne recenzje')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "popularity_vs_quality_bubble.png"))
        plt.close()
        
    # 6. Statistical Summary Visualization
    if "review_length" in sentiment_length_pd.columns:
        stats_data = sentiment_length_pd['review_length'].describe()
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(stats_data)), stats_data.values, 
                      color=['skyblue', 'lightgreen', 'orange', 'red', 'purple', 'brown', 'pink', 'gray'])
        plt.title('Statystyki opisowe długości recenzji')
        plt.xlabel('Statystyki')
        plt.ylabel('Wartość')
        plt.xticks(range(len(stats_data)), stats_data.index, rotation=45)
        
        for bar, value in zip(bars, stats_data.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stats_data.values)*0.01, 
                    f'{value:.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "review_length_statistics.png"))
        plt.close()

    print("\n[SUCCESS] Wszystkie dodatkowe wykresy zostały wygenerowane!")
    print("Nowe wykresy:")
    print("- Rozkład długości recenzji (histogram)")
    print("- Top 20 gier według liczby recenzji")
    print("- Długość recenzji według sentymentu (box plot)")
    print("- Rozkład sentymentu (wykres kołowy)")
    print("- Popularność vs jakość gier (wykres bąbelkowy)")
    print("- Statystyki opisowe długości recenzji")

@timed("Save Output Data")
def save_outputs(output_dir: str,
                 rdd_counts_pd: pd.DataFrame, rdd_avg_pd: pd.DataFrame, rdd_sentiment_pd: pd.DataFrame,
                 df_counts: DataFrame, df_avg: DataFrame, df_sentiment: DataFrame,
                 top_games_pd: pd.DataFrame, score_distribution_pd: pd.DataFrame, app_avg_pd: pd.DataFrame) -> None:
    os.makedirs(output_dir, exist_ok=True)

    rdd_counts_pd.to_csv(os.path.join(output_dir, "rdd_review_counts.csv"), index=False)
    rdd_avg_pd.to_csv(os.path.join(output_dir, "rdd_avg_scores.csv"), index=False)
    rdd_sentiment_pd.to_csv(os.path.join(output_dir, "rdd_sentiment_counts.csv"), index=False)

    df_counts.toPandas().to_csv(os.path.join(output_dir, "df_review_counts.csv"), index=False)
    df_avg.toPandas().to_csv(os.path.join(output_dir, "df_avg_scores.csv"), index=False)
    df_sentiment.toPandas().to_csv(os.path.join(output_dir, "df_sentiment_counts.csv"), index=False)

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
    length_pd, word_counts_pd, year_pd, month_pd, sentiment_length_pd, games_sentiment_pd, correlation_df = extended_eda(
        cleaned_df,
        os.path.join(os.path.dirname(os.path.dirname(__file__)), OUTPUT_DIR),
        threshold
    )

    generate_extended_visualizations(
        length_pd, word_counts_pd, year_pd, month_pd, sentiment_length_pd, games_sentiment_pd, correlation_df,
        os.path.join(os.path.dirname(os.path.dirname(__file__)), OUTPUT_DIR)
    )
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
