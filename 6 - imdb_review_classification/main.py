# importing the libraries
import tensorflow as tf
import keras
from keras import models
from keras import layers
from keras.datasets import imdb
import numpy as np
import matplotlib.pyplot as plt

# printing the version of tensorflow
print(f"tensorflow version : {tf.__version__}")

# loading the data
(train_text , train_label) , (test_text , test_label) = imdb.load_data(num_words = 10000)
print(f"Train Text Shape : {train_text.shape}")
print(f"Train Label Shape : {train_label.shape}")
print(f"Test Text Shape : {test_text.shape}")
print(f"Test Label Shape : {test_label.shape}")

# making the vector hot encoded
def vectorize_sequence(sequence , dimensions = 10000):
    zero_arr = np.zeros((len(sequence) , dimensions))
    for i , seq in enumerate(sequence):
        zero_arr[i , seq] = 1
    return zero_arr
train_text = vectorize_sequence(train_text)
test_text = vectorize_sequence(test_text)
train_label = np.asarray(train_label).astype("float32")
test_label = np.asarray(test_label).astype("float32")

# making the validation_dataset
val_text = test_text[:20000]
val_label = test_label[:20000]
test_text = test_text[20000:]
test_label = test_label[20000:]

# making the model
model = models.Sequential()
model.add(layers.Dense(32 , activation = 'relu' , input_shape = (10000 , )))
model.add(layers.Dense(32 , activation = 'relu'))
model.add(layers.Dense(1 , activation = 'sigmoid'))

# compiling the model
model.compile(
    optimizer = 'adam' , 
    loss = 'binary_crossentropy' , 
    metrics = ['accuracy'] 
)

# fitting the model
history = model.fit(train_text , train_label , epochs = 10 , validation_data = (val_text , val_label) , batch_size = 128)

# plotting the graph 
plt.figure(figsize = (16 , 8))

plt.subplot(1 , 2 , 1)
plt.plot(history.history['accuracy'] , label = "accuracy")
plt.plot(history.history['val_accuracy'] , label = "validation accuracy")
plt.legend(loc = "lower right")
plt.title("Accuracy")
plt.xlabel("EPOCHS")

plt.subplot(1 , 2 , 2)
plt.plot(history.history['loss'] , label = "loss")
plt.plot(history.history['val_loss'] , label = "validation loss")
plt.legend(loc = "upper right")
plt.title("Loss")
plt.xlabel("EPOCHS")

plt.show()

# evaluating the model
predictions = model.predict(test_text)
while(True):
    i = int(input("Enter the string : "))
    print(predictions[i])
    print(test_label[i])