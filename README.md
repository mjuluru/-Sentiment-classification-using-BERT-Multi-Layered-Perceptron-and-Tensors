# Sentiment-classification-using-BERT-Multi-Layered-Perceptron-and-Tensors
Machine Learning model using Python to perform sentiment analysis on Twitter data

**Dataset:**

The dataset used is a Sentiment140 dataset (https://www.kaggle.com/kazanova/sentiment140) containing 1,600,000 tweets. I used a sample of 80,000 
entries and their targets. Targets used in the dataset are 0 for Negative tweet and 4 for Positive tweet.
The dataset is preprocessed to drop the other columns, namely, ids, date, flag, and user, which weren’t 
required for sentiment analysis and binarized the dataset into 0 for negative tweets and 1 for positive 
tweets. I have also removed the noise such as names and special characters in tweets to smoothen the 
process of sentiment analysis.

I split the sampled dataset using train_test_split() function into train, val and test sets for training, 
valuation and testing the model with rows ~ 35000, 15000 and 30000 respectively.

**List of Parameters used in the code:**

Learning rate used for the sentiment analysis is 0.0003. The Multi-Layer Perceptron uses a feed-forward 
classifier with three Linear layers and two activation functions ReLU() and Sigmoid(). The batch size 
of 32 is used to fine tune BERT and a maximum length of 45 is used for tokenization.

**Results:**

I have derived BERT encodings for the train and valuation sets and sent them as inputs to train the 
three-layered Multi-Layered Perceptron. The results obtained from the sentiment analysis for each 
epoch is as follows:
EPOCH   Train Accuracy  Valuation Accuracy
1       73.828          75.957
2       76.230          77.910
3       76.480          77.938
4       76.895          78.109
5       77.277          78.324

Overall Test accuracy = 78.28%

Challenges faced during the homework include memory shortage while using a batch size of 64 or more, 
google collab restriction of GPU time per account per day and runtime disconnections while executing 
the code.
