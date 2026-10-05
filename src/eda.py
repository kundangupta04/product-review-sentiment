
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")  # save figures to files without opening windows
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

INPUT_PATH = "data/processed/reviews_cleaned.csv"
FIG_DIR = "outputs/figures"
COLORS = {"Positive": "#2e8b57", "Negative": "#c0392b"}
ORDER = ["Positive", "Negative"]

os.makedirs(FIG_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")

df = pd.read_csv(INPUT_PATH)

# Review length = number of words in the cleaned review
df["review_length"] = df["clean_review"].str.split().str.len()

# ------------------------------------------------------------
# 1. Summary statistics
# ------------------------------------------------------------
print("=" * 60)
print("SUMMARY STATISTICS")
print("Total reviews    :", len(df))
print("Positive reviews :", (df["sentiment"] == "Positive").sum())
print("Negative reviews :", (df["sentiment"] == "Negative").sum())
print(f"Average length   : {df['review_length'].mean():.2f} words")
print("Minimum length   :", df["review_length"].min(), "words")
print("Maximum length   :", df["review_length"].max(), "words")
print("Median length    :", df["review_length"].median(), "words")

print("\nAverage length by sentiment (words):")
print(df.groupby("sentiment")["review_length"].mean().round(2))

# ------------------------------------------------------------
# 2. Chart 1: Positive vs Negative count
# ------------------------------------------------------------
plt.figure(figsize=(6, 4))
ax = sns.countplot(data=df, x="sentiment", order=ORDER, hue="sentiment",
                   palette=COLORS, legend=False)
for container in ax.containers:
    ax.bar_label(container)  # show the count on top of each bar
plt.title("Positive vs Negative Review Count")
plt.xlabel("Sentiment")
plt.ylabel("Number of reviews")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/class_distribution.png", dpi=150)
plt.close()
print("\nSaved: class_distribution.png")

# ------------------------------------------------------------
# 3. Chart 2: Review length distribution
#    (x-axis is cut at the 99th percentile so a few very long
#     reviews do not squash the chart)
# ------------------------------------------------------------
limit = df["review_length"].quantile(0.99)
plt.figure(figsize=(8, 4))
sns.histplot(data=df[df["review_length"] <= limit], x="review_length",
             bins=50, color="#2c7fb8")
plt.title("Review Length Distribution (words, up to 99th percentile)")
plt.xlabel("Words per review")
plt.ylabel("Number of reviews")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/review_length_distribution.png", dpi=150)
plt.close()
print("Saved: review_length_distribution.png")

# ------------------------------------------------------------
# 4. Chart 3 (optional): Most frequent words per sentiment
# ------------------------------------------------------------
def top_words(texts, n=15):
    """Count the most common words, ignoring common filler words."""
    counter = Counter()
    for text in texts:
        for word in text.split():
            if len(word) > 2 and word not in ENGLISH_STOP_WORDS:
                counter[word] += 1
    return counter.most_common(n)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, label in zip(axes, ORDER):
    words = top_words(df.loc[df["sentiment"] == label, "clean_review"])
    names = [w for w, _ in words]
    counts = [c for _, c in words]
    sns.barplot(x=counts, y=names, ax=ax, color=COLORS[label])
    ax.set_title(f"Top 15 Words in {label} Reviews")
    ax.set_xlabel("Frequency")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/top_words.png", dpi=150)
plt.close()
print("Saved: top_words.png")

# ------------------------------------------------------------
# 5. Charts 4 and 5 (optional): Word clouds
# ------------------------------------------------------------
try:
    from wordcloud import WordCloud

    for label in ORDER:
        text = " ".join(df.loc[df["sentiment"] == label, "clean_review"])
        cloud = WordCloud(width=900, height=450, background_color="white",
                          stopwords=set(ENGLISH_STOP_WORDS),
                          random_state=42).generate(text)
        plt.figure(figsize=(9, 4.5))
        plt.imshow(cloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"{label} Reviews Word Cloud")
        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}/wordcloud_{label.lower()}.png", dpi=150)
        plt.close()
        print(f"Saved: wordcloud_{label.lower()}.png")
except ImportError:
    print("wordcloud not installed, so word clouds were skipped (optional).")

print("\nEDA finished. Check the folder:", FIG_DIR)