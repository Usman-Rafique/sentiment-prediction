'''
A convolutional neural network for sentiment classification using IMDB dataset.
The network gives over 88% accuracy after 30 epochs
This project is for educational purposes and without warranty of any kind.


Author: M. Usman Rafique
'''

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import Dropout, BatchNormalization, SpatialDropout1D, Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop, Adam
from keras import regularizers

import matplotlib.pyplot as plt

max_features = 20000
max_length = 250  # Read only 250 words of every review
batch_size = 32
epochs = 30

#Load the dataset using custom max. length
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

x_train = sequence.pad_sequences(x_train, maxlen=max_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_length)

# Used for visualizing the data
print('x_train[0,:] = ',x_train[0,:])
print('y_train[0] = ', y_train[0])

model = Sequential()
model.add(Embedding(max_features, 16, input_length=max_length, embeddings_initializer='glorot_uniform'))
model.add(SpatialDropout1D(0.2))


model.add(Conv1D(5, 3, padding='valid', activation='linear', strides=1, kernel_initializer='glorot_uniform'))
model.add(LeakyReLU(0.1))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv1D(15, 3, padding='valid', activation='linear', strides=1, kernel_initializer='glorot_uniform'))
model.add(LeakyReLU(0.1))
model.add(Dropout(0.2))

model.add(Conv1D(15, 3, padding='valid', activation='linear', strides=1, kernel_initializer='glorot_uniform'))
model.add(LeakyReLU(0.1))
model.add(Dropout(0.2))

model.add(GlobalMaxPooling1D())

model.add(Dense(50, activation='linear', kernel_initializer='glorot_uniform'))
model.add(LeakyReLU(0.1))
model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))

# Compiling the model
model.compile(loss='binary_crossentropy',
              optimizer= RMSprop(),
              metrics=['accuracy'])

# Call backs to improve learning
reduce_lr = ReduceLROnPlateau(factor=0.2, patience=2, min_lr=0.00001)
myCheckPoint = ModelCheckpoint(filepath = 'usman_best_CNN.hdf5', monitor='val_acc',save_best_only=True)
earlyStopping = EarlyStopping(patience=8)

model.summary()

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          callbacks=[myCheckPoint, reduce_lr, earlyStopping])

# load the best model
model.load_weights('usman_best_CNN.hdf5')


# Evaulate the model
test_loss, test_acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score (best model):', test_loss)
print('Test accuracy (best model):', test_acc)

# Plot the convergence curves of loss and accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Testing'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Testing'], loc='upper left')
plt.show()

