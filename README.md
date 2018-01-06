# Sentiment Prediction from Movie Reviews on IMDB

The goal of this project is to understand the sentiment from a given text.

This project uses the IMDB dataset for sentiment analysis. The dataset contains movie reveiws and every review has been labelled as either positive or negative. The problem is to correctly predict if a review is positie or negative. For more details about the dataset, please see: http://ai.stanford.edu/~amaas/data/sentiment/.

First, we need to convert the words to vectors using an embedding layer. Since every word is originally represented as an integer, we need to convert them to vectors of real numbers so that vectors of similar words are "closer". You can read more about embedding layer in Keras here: https://keras.io/layers/embeddings/.

## Key Features
This network is not designed for the best accuracy. Instead, the goal is to make an efficient network which can do well while using minimun bumber of parameters. One way to quantify the score that includes test accuracy and the number of paramaters of the network is:

![Score equation](https://github.com/Usman-Rafique/sentiment-prediction/blob/master/score.png)

[//]: # ( Score = \frac{Accuracy}{(Number of Parameters)^0.1}  )

## Accuracy vs Epochs
![Accuracy Plot](https://github.com/Usman-Rafique/sentiment-prediction/blob/master/CNN_acc.png)

## Loss vs Epochs
![Loss Plot](https://github.com/Usman-Rafique/sentiment-prediction/blob/master/CNN_loss.png)



