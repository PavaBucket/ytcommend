import collect
import treat
import clean
import firstModel
import actLearning
import actLearningTest
import mainModel

import variables

# Initializing variables
variables.init()

# collect.collect()
# treat.treat()
clean.clean()
firstModel.firstModel()
# actLearning.actLearning(mlData)
actLearningTest.actLearningTest()
mainModel.mainModel()
