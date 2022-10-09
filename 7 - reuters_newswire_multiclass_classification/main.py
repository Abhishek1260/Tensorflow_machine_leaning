# importing the libraries
import tensorflow as tf
import keras
from keras import models
from keras import layers
from keras.datasets import reuters
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

# printing the version of tensorflow
print(f"Tensorflow Version : {tf.__version__}")

# getting the data
(train_data , train_label) , (test_data , test_label) = reuters.load_data(num_words = 10000)
print(f"Train Data Shape : {train_data.shape}")
print(f"Train Label Shape : {train_label.shape}")
print(f"Test Data Shape : {test_data.shape}")
print(f"Test Label Shape : {test_label.shape}")

# seeing the data
print(f"Train Data at 10 : {train_data[10]}")
print(f"Train Label at 10 : {train_label[10]}")

# making the data usable
def vectorize_sequence(sequence , dimensions = 10000):
    result = np.zeros((len(sequence) , dimensions))
    for i , seq in enumerate(sequence):
        result[i , seq] = 1
    return result
train_data = vectorize_sequence(train_data)
test_data = vectorize_sequence(test_data)
train_label = to_categorical(train_label)
test_label = to_categorical(test_label)

# making the model ready for the classification
model = models.Sequential()
model.add(layers.Dense(64 , activation = 'relu' , input_shape = (10000, )))
model.add(layers.Dense(64 , activation = 'relu'))
model.add(layers.Dense(46 , activation = 'softmax'))

# compiling the model
model.compile(
    loss = "categorical_crossentropy" , 
    optimizer = "adam" , 
    metrics = ['accuracy']
)

# making the validation set
val_data = test_data[:1800]
val_label = test_label[:1800]
test_data = test_data[1800:]
test_label = test_label[1800:]

# fitting the model
history = model.fit(
    train_data , 
    train_label , 
    epochs = 10 , 
    validation_data = (val_data , val_label) , 
    batch_size = 256
)

# plotting the graph
plt.figure(figsize = (16 , 8))

plt.subplot(1 , 2 , 1)
plt.plot(history.history['accuracy'] , label = "accuracy")
plt.plot(history.history['val_accuracy'] , label = 'validation_accuracy')
plt.legend(loc = "lower right")
plt.title("ACCURACY")
plt.xlabel("EPOCHS")

plt.subplot(1 , 2 , 2)
plt.plot(history.history['loss'] , label = "loss")
plt.plot(history.history['val_loss'] , label = "validation loss")
plt.legend(loc = "upper right")
plt.title("LOSS")
plt.xlabel("EPOCHS")

plt.show()

# predicting the variableds
prediction = model.predict(test_data)
while(True):
    i = int(input("Enter the string : "))
    print(prediction[i])
    print(test_label[i])