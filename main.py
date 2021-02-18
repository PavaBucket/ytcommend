import collect
import treat
import clean
import firstModel
import actLearning
import actLearningTest

# collect.collect()
# treat.treat()
mlData = clean.clean()
mlData = firstModel.firstModel(mlData)
# actLearning.actLearning(mlData)
actLearningTest.actLearningTest(mlData)
