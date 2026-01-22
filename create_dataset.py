import pandas as pd

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 0
true["label"] = 1

df = pd.concat([fake, true], ignore_index=True)

df["text"] = df["title"] + " " + df["text"]

df = df[["text", "label"]]

df.to_csv("data/news.csv", index=False)

print("news.csv created successfully!")
