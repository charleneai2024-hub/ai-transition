import pandas as pd

df = pd.read_csv("data/sentimentdataset.csv")
df["Sentiment"] = df["Sentiment"].str.strip()

positive = {"Positive","Happiness","Joy","Love","Amusement","Enjoyment",
            "Admiration","Affection","Awe","Surprise","Acceptance","Adoration",
            "Anticipation","Excitement","Gratitude","Contentment","Hope","Pride",
            "Elation","Euphoria","Serenity"}
negative = {"Negative","Anger","Fear","Sadness","Disgust","Disappointed",
            "Disappointment","Frustration","Loneliness","Grief","Boredom",
            "Anxiety","Despair","Shame","Regret"}

mapped = positive | negative
unmapped = df[~df["Sentiment"].isin(mapped)]["Sentiment"].value_counts()
print("Number of unmapped labels:", unmapped.shape[0])
print("\nTop 30 unmapped labels (count):\n", unmapped.head(30))
