# NLP-Pipeline
This repository contains the files/programs used as part of a NLP pipeline consisting of data cleaning and model training/prediction

## Installations
The Anaconda distribution is required to run the code in this repository.

The necessary libraries to run the .py scripts are: 
- pandas
- numpy
- nltk
- scikit-learn

The version of python used for running the .py scripts is Python 3.6.3. 

In addition to the above libraries, the following libraries are needed to run the .ipynb file:
-xgboost
-hyperopt

The version of python used for the analysis/model development is Python 3.10.9. 

## Motivation

The pipeline created and analysis/model development conducted in this repository was part of the second project in the Udacity Data Science Nanodegree.

The objective of this pipeline is to:
1. Read in text in the form of private messages, public social media posts and news related to disaster events
2. Classify these messages based on 36 pre-defined categories. Messages can fall into multiple categories

## Files in this repository

### process_data.py

This script takes in the files data/disaster_messages.csv and data/disaster_categories.csv, merges and cleans the data, then outputs the processed data to a .db file.

To run this script: python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

### train_classifier.py

This script reads in the .db file generated by process_data.py, trains the model, then outputs the model to a .pkl file.

To run this script: python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

### run.py

This script creates a local web app showcasing the results of the pipeline.

### model_tuning.ipynb

Due to the poor performance of the random forest classifier, I explored the xgboost classifier as a possible alternative. While the xgboost classifier performed much better than the random forest classifier, I was unable to sucessfully implement it, as training on the web IDE provided by Udacity for this project took far too long. 
