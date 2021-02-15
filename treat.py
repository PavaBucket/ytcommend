import pandas as pd


def treat():
    ytDataFrame = pd.read_json("./data/ytRawLinks.json", lines=True)
    ytTreatedDataFrame = ytDataFrame.drop_duplicates(subset=['webpage_url'])
    ytTreatedDataFrame.to_csv("./data/ytTreatedLinks.csv")
