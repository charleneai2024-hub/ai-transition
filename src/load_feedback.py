import pandas as pd

# --- Load ---
df = pd.read_csv("data/sentimentdataset.csv")
print("Raw shape:", df.shape)

# --- 1. Drop junk index columns ---
df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])

# --- 2. Strip whitespace from text columns ---
for col in ["Text", "Sentiment", "Platform", "Country", "User"]:
    df[col] = df[col].str.strip()

# --- 3. Collapse fine-grained labels into 3 buckets ---
positive = {"Positive","Happiness","Happy","Joy","Love","Amusement","Enjoyment",
            "Admiration","Affection","Awe","Surprise","Acceptance","Adoration",
            "Anticipation","Excitement","Gratitude","Contentment","Hope","Hopeful",
            "Pride","Proud","Elation","Euphoria","Serenity","Enthusiasm",
            "Determination","Playful","Inspiration","Inspired","Empowerment",
            "Calmness","Compassion","Tenderness","Fulfillment","Reverence","Arousal"}
negative = {"Negative","Anger","Fear","Sadness","Sad","Disgust","Disappointed",
            "Disappointment","Frustration","Frustrated","Loneliness","Grief",
            "Boredom","Anxiety","Despair","Shame","Regret","Confusion","Embarrassed",
            "Numbness","Melancholy","Hate","Bad","Bitterness","Betrayal"}

def bucket(label):
    if label in positive: return "positive"
    if label in negative: return "negative"
    return "neutral"

df["sentiment_3"] = df["Sentiment"].apply(bucket)

# --- 4. Keep reasonable-length text for the LLM later ---
df = df[df["Text"].str.len().between(5, 500)]

# --- Inspect ---
print("Clean shape:", df.shape)
print("\n3-class distribution:\n", df.groupby("sentiment_3").size())

# --- Save clean output for the flagship project ---
df[["Text", "sentiment_3", "Platform", "Likes", "Retweets"]].to_parquet(
    "data/feedback_clean.parquet"
)
print("\nSaved -> data/feedback_clean.parquet")
