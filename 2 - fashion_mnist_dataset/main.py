# importing the stuff for the working
import tensorflow as tf
import keras
from keras import models
from keras.utils import to_categorical
from keras import layers
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

# printing the version of tensorflow
print(f"Tensorflow Version : {tf.__version__}")

# getting the data
(train_images , train_labels) , (test_images , test_labels) = fashion_mnist.load_data()
print(f"Train Images Shape : {train_images.shape}")
print(f"Train Labels Shape : {train_labels.shape}")
print(f"Test Images Shape : {test_images.shape}")
print(f"Test Labels Shape : {test_labels.shape}")

# making the data normalized
train_images = train_images / 255
test_images = test_images / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# making the model
model = models.Sequential()
model.add(layers.Flatten(input_shape = (28 , 28 , 1)))
model.add(layers.Dense(512 , activation = 'relu'))
model.add(layers.Dense(10 , activation = 'softmax'))

# compiling the model
model.compile(
    optimizer = 'adam' , 
    loss = "categorical_crossentropy" ,
    metrics = ['accuracy']
)

# fitting the model
model.fit(train_images , train_labels , epochs = 10)

# predicting from the model
predictions = model.predict(
    test_images
)
plt.figure()
while(True):
    i = int(input("Enter the number to see : "))
    plt.imshow(test_images[i])
    plt.show()
    print(predictions[i])
    print(test_labels[i])