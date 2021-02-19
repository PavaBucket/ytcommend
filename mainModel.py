import optimization
import settings


def mainModel(mlData):
    optimization.optimizeLGBM(mlData, settings.lgbmParameters, settings.tfidfParameters)

    return mlData
