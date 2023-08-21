# NBA Runs Prediction model

A novel approach attempting to predict the likelihood that a basketball team will achieve a scoring run based on events precipitating the start of the run. The model utilizes convolutional neural network architecture in order to classify a set of basketball events as capable of leading to a scoring run. The model has been trained on five NBA season, from 2015 to 2020, and has been shown to achieve 65% accuracy, outperforming conventional classifiers like Support Vector Classifier, Decision Tree Classifier, Random Forests Classifier, and k-Nearest Neighbors Classifier. The model has been developed on the Keras API and given a set of 10 events over the attributes listed below, the model can provide a prediction on whether those 10 events can lead to a run or not. Simply load the model and use the predict() function. Alongside the model, the a notebook is provided explaining the development process.

### Selected Features

- Quarter
- ShotType
- ShotDist
- FoulType
- TimeoutTeam
- EnterGame
- LeaveGame
- Shooter
- Rebounder
- Blocker
- Fouler
- Fouled
- ReboundType
- ViolationPlayer
- ViolationType
- FreeThrowShooter
- TurnoverPlayer
- TurnoverType
- TurnoverCause
- TurnoverCauser
