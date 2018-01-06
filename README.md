# Sentiment Prediction from Movie Reviews on IMDB

The goal of this project is to understand the sentiment from a given text.

This project uses the **Large Movie Review Dataset** (also known as IMDB dataset) for sentiment analysis. The dataset contains movie reveiws - every review has been labelled as either positive or negative. The problem is to correctly predict if a review is positie or negative. For more details about the dataset, please see: http://ai.stanford.edu/~amaas/data/sentiment/.

To make the problem (slightly) harder, we only read 250 words from every review. The average review size is around 300 words and some commonly reffered examples use first 500 words from every review.

## Network Architecture

First, we need to convert the words to vectors using an embedding layer. Since every word is originally represented as an integer, we need to convert them to vectors of real numbers so that vectors of similar words are "closer". You can read more about embedding layer in Keras here: https://keras.io/layers/embeddings/.

## Key Features
This network is not designed for the best accuracy. Instead, the goal is to make an efficient network which can do well while using minimun bumber of parameters. One way to quantify the score that includes test accuracy and the number of paramaters of the network is:

![Score equation](https://github.com/Usman-Rafique/sentiment-prediction/blob/master/score.png)

This network gives a test accuracy of over 88% (88.376% in my last test). Since there are 322,046 total parameters in the network, we get **score = 0.2486** which is not too bad.

**Network Summary** *same paragraph*
Total number of parameters: 322,046 *same paragraph*
Test Accuracy: 88%m *same paragraph*
Training Accuracy: 91%
Convolutional Layers: 3

## Secret Sauce
Yes, it is not a secret anymore! The key decision that makes the number of parameters small, without compromising the accuracy, is to reduce the output size of embedding layer. Typical value of 132-dimensional output vector generates millions of parameters in just the embedding layer! While I have reduced the number of feature maps in the convolutional layers and the fully connected layer, the major improvement comes from reducing the dimension of vectors by the embedding layer.


## Accuracy vs Epochs
![Accuracy Plot](https://github.com/Usman-Rafique/sentiment-prediction/blob/master/CNN_acc.png)

## Loss vs Epochs
![Loss Plot](https://github.com/Usman-Rafique/sentiment-prediction/blob/master/CNN_loss.png)



