# NBA Runs Prediction model

A novel approach attempting to predict the likelihood that a basketball team will achieve a scoring run based on events precipitating the start of the run. The model utilizes convolutional neural network architecture in order to classify a set of basketball events as capable of leading to a scoring run. The model has been trained on five NBA season, from 2015 to 2020, and has been shown to achieve 63% accuracy, outperforming conventional classifiers like Support Vector Classifier, Decision Tree Classifier, Random Forests Classifier, and k-Nearest Neighbors Classifier. The model has been developed on the Keras API and given a set of 10 events over the attributes listed below, the model can provide a prediction on whether those 10 events can lead to a run or not. 

Included is the pre-trained model, which can be used by running the predictions.py file. The model can perform predictions for any dataset scrapped from basketball-reference.com. Simply preprocess the dataset using the corresponding file. A new model can also be trained by running the model_training.py file on any dataset. Included is also a notebook that explains the development of the model and investigates the player and team decisions that affect the emergence of a scoring run. 
