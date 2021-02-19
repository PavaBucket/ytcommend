import functions
import pandas as pd
import settings


def treat():
    ytDataFrame = pd.read_json(settings.rawLinksPath, lines=True)
    ytTreatedDataFrame = functions.treat(ytDataFrame)
    ytTreatedDataFrame.to_csv(settings.treatedLinksPath)
