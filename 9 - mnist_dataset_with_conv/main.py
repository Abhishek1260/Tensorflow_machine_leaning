# importing the data
import tensorflow as tf
import keras 
from keras import models
from keras import layers
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# printing the version of tensorflow
print(f"tensorflow version : {tf.__version__}")

# getting the data
(train_data , train_label) , (test_data , test_label) = mnist.load_data()
print(f"Train Data shape : {train_data.shape}")
print(f"Train Label shape : {train_label.shape}")
print(f"Test Data shape : {test_data.shape}")
print(f"Test Label shape : {test_label.shape}")

# normalizing the data
train_data = train_data / 255
test_data = test_data / 255
train_label = to_categorical(train_label)
test_label = to_categorical(test_label)

# making the validation set
val_data = test_data[:8000]
val_label = test_label[:8000]
test_data = test_data[8000:]
test_label = test_label[:8000]

# making the model
model = models.Sequential()
model.add(layers.Conv2D(32 , (3 , 3) , input_shape = (28 , 28 , 1) , activation = 'relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(64 , (3 , 3) , activation = 'relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Flatten())
model.add(layers.Dense(256 , activation = 'relu'))
model.add(layers.Dense(10 , activation = 'softmax'))

# printing the model summary
model.summary()

# compiling the model
model.compile(
    optimizer = "adam" , 
    loss = "categorical_crossentropy" , 
    metrics = ['accuracy']
)

# fitting the model
history = model.fit(train_data , train_label , epochs = 10 , validation_data = (val_data , val_label))

# plotting the graph
plt.figure(figsize = (16,  8))

plt.subplot(1 , 2 , 1)
plt.plot(history.history['accuracy'] , label = 'accuracy')
plt.plot(history.history['val_accuracy'] , label = 'validation accuracy')
plt.title("ACCURACY")
plt.xlabel("EPOCHS")
plt.legend(loc = 'lower right')

plt.subplot(1 , 2 , 2)
plt.plot(history.history['loss'] , label = 'loss')
plt.plot(history.history['val_loss'] , label = 'validation loss')
plt.title("LOSS")
plt.xlabel("EPOCHS")
plt.legend(loc = "upper right")

plt.show()

# predicting the model
predictions = model.predict(test_data)
while(True):
    i = int(input("Enter the number : "))
    print(predictions[i])
    print(test_label[i])