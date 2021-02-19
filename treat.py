import functions
import pandas as pd


def treat():
    ytDataFrame = pd.read_json("./data/ytRawLinks.json", lines=True)
    ytTreatedDataFrame = functions.treat(ytDataFrame)
    ytTreatedDataFrame.to_csv("./data/ytTreatedLinks.csv")
