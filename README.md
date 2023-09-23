# NBA Runs Prediction model

A novel approach attempting to predict the likelihood that a basketball team will achieve a scoring run based on events precipitating the start of the run. The model utilizes convolutional neural network architecture in order to classify a set of basketball events as capable of leading to a scoring run. The model has been shown to achieve 63% accuracy, outperforming conventional classifiers like Support Vector Classifier, Decision Tree Classifier, Random Forests Classifier, and k-Nearest Neighbors Classifier. 

## Data & Pre-processing

The play-by-play data used for training the model has been scraped from basketball-reference.com and covers 5 NBA seasons, from 2015 to 2020. The model takes into account 20 features in total, describing in-game events. The data has been encoded and split into sets of 10 consecutive events that either led to a scoring run by a team or did not. Each set was then reshaped into a 20x10 matrix for training. File preprocessing.py can be used to replicate the process and prepare datasets for either predictions or training.

## Model architecture

A convolutional neural network was developed in order to perform predictions on whether a set of 10 consecutive events can lead to a scoring run. The pretrained model can be loaded and be used on any datasets preprocessed in the manner described above. Moreover, the model_training.py can be used to train models on different datasets. 

![image](https://github.com/isevr/NBA_runs_prediction/assets/114408865/799cf284-89fe-4333-9577-d7b4e9fa07aa)

## Notebook

A notebook file is included that describes the development process for the neural network. Furthermore, the notebook underlines specific players, teams, actions, and decisions that mostly affect the probability of a scoring run occuring.

