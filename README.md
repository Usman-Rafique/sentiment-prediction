## Sentiment Prediction from Movie Reviews on IMDB

The goal of this project is to understand the sentiment from a given text.

This project uses the IMDB dataset for sentiment analysis. The dataset contains movie reveiws and every review has been labelled as either positive or negative. The problem is to correctly predict if a review is positie or negative. For more details about the dataset, please see: http://ai.stanford.edu/~amaas/data/sentiment/.

First, we need to convert the words to vectors using an embedding layer. Since every word is originally represented as an integer, we need to convert them to vectors of real numbers so that vectors of similar words are "closer". You can read more about embedding layer in Keras here: https://keras.io/layers/embeddings/.


# Accuracy vs Epochs
![Accuracy Plot](https://github.com/Usman-Rafique/sentiment-prediction/blob/master/CNN_acc.png)

# Loss vs Epochs
![Loss Plot](https://github.com/Usman-Rafique/sentiment-prediction/blob/master/CNN_loss.png)



