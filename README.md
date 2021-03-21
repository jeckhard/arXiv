# arXiv

Tools to analyse the arXiv data set supporting dense neural networks, XGBoost, random forest and SVMs

In order to use the models in Models.py it is necessary to

- import the 'arxiv-metadata.json' dataset from https://www.kaggle.com/Cornell-University/arxiv
- run CleanData.py to clean the data
- run EncodeData.py  to encode the data using the universal sentence encoder

Analysis of the dataset to determine the category, one of {'physics','cs','math'}
Hyperparameter tuning for all models and different training set sizes
