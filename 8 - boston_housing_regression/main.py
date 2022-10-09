# importing the libraries
import tensorflow as tf
import keras
from keras import models
from keras import layers
import numpy as np
from keras.datasets import boston_housing
import matplotlib.pyplot as plt

# printing the tensorflow
print(tf.__version__)

# downloading the dataset
(train_data , train_label) , (test_data , test_label) = boston_housing.load_data()
print(f"Train Data Shape : {train_data.shape}")
print(f"Train Label Shape : {train_label.shape}")
print(f"Test Data Shape : {test_data.shape}")
print(f"Test Label Shape : {test_label.shape}")

# making the data normalized
val_data = test_data[:80]
val_label = test_label[:80]
test_data = test_data[80:]
test_label = test_label[80:]
mean = train_data.mean(axis = 0)
train_data -= mean
stf = train_data.std(axis = 0)
train_data /= stf
test_data -= mean
test_data /= stf
val_data -= mean
val_data /= stf

# making the validation set

# makiing the model
model = models.Sequential()
model.add(layers.Dense(64 , activation = 'relu' , input_shape = (train_data.shape[1] ,)))
model.add(layers.Dense(128 , activation = 'relu'))
model.add(layers.Dense(1))

# compiling the model
model.compile(
    loss = "mse" , 
    optimizer = "rmsprop" , 
    metrics = ['mae']
)

# fitting the model
history = model.fit(train_data , train_label , epochs = 100 , validation_data = (val_data , val_label))

# plotting the history graph
plt.figure(figsize = (16 , 8))

plt.subplot(1 , 2 , 1)
plt.plot(history.history["mae"] , label = "mae")
plt.plot(history.history['val_mae'] , label = "val_mae")
plt.legend(loc = "upper right")
plt.xlabel("EPOCHS")
plt.title("MAE")      
 
plt.subplot(1 , 2 , 2)
plt.plot(history.history["loss"] , label = "loss")
plt.plot(history.history['val_loss'] , label = "val_loss")
plt.legend(loc = "upper right")
plt.xlabel("EPOCHS")
plt.title("LOSS")

plt.show()

# making the predictions
predictions = model.predict(test_data)
for i in range(len(predictions)):
    print(f"prediction -> {predictions[i]} : Real -> {test_label[i]}")