# importing the tensorflow
import tensorflow as tf
import keras
from keras import models
from keras import layers
from keras.utils import to_categorical
from keras.datasets import mnist
import matplotlib.pyplot as plt

# printing the version of tensorflow
print(f"Tensorflow version : {tf.__version__}")

# making the plot
plt.figure()

# getting the data
(train_images , train_labels) , (test_images , test_labels) = mnist.load_data()
print(f"Train Image Shape : {train_images.shape}")
print(f"Train Labels Shape : {train_labels.shape}")
print(f"Test Image Shape : {test_images.shape}")
print(f"Test Labels Shape : {test_labels.shape}")

# making the image data in working condition
train_images = train_images / 255
test_images = test_images / 255
train_labels = to_categorical(train_labels)
test_label = to_categorical(test_labels)

# making the model
model = models.Sequential()
model.add(layers.Flatten(input_shape = (28 , 28 , 1)))
model.add(layers.Dense(512 , activation='relu'))
model.add(layers.Dense(10 , activation='softmax'))

# compiling the models
model.compile(optimizer = "rmsprop" , loss = "categorical_crossentropy" , metrics = ['accuracy'])

# fitting the model
model.fit(train_images , train_labels , epochs = 10)

# predicting the output
prediction = model.predict(test_images)
while(True):
    i = int(input("enter the number to predicts : "))
    plt.imshow(test_images[i])
    plt.show()
    print(prediction[i])
    print(test_label[i])