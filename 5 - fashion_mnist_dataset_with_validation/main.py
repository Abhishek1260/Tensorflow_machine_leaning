# importing the libraries
from cgi import test
import tensorflow as tf
import keras
from keras import layers
from keras import models
from keras.utils import to_categorical
from keras.datasets import fashion_mnist

# printing the version of tensorflow
print(f"Tensorflow Version : {tf.__version__}")

# downloading the dataset into the training and testing dataset
(train_images , train_labels) , (test_images , test_labels) = fashion_mnist.load_data()
print(f'Train Image Shape : {train_images.shape}')
print(f'Train Label Shape : {train_labels.shape}')
print(f'Test Image Shape : {test_images.shape}')
print(f'Test Label Shape : {test_labels.shape}')

# making the validation dataset
validation_images = test_images[:8000]
validation_labels = test_labels[:8000]
test_images = test_images[8000:]
test_labels = test_labels[8000:]

# making the dataset available for training, validation, and prediction
train_images = train_images / 255
test_images = test_images / 255
validation_images = validation_images / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
validation_labels = to_categorical(validation_labels)

# making the model
model = models.Sequential()
model.add(layers.Flatten(input_shape = (28 , 28 , 1)))
model.add(layers.Dense(512 ,activation = 'relu'))
model.add(layers.Dense(10 , activation = 'softmax'))

# compiling the model
model.compile(
    optimizer = 'adam' , 
    loss = 'categorical_crossentropy' , 
    metrics = ['accuracy']
)

# fitting the model
model.fit(train_images , train_labels , validation_data = (validation_images , validation_labels) , epochs = 10)

# predicting from the model
predictions = model.predict(test_images)
while (True):
    i = int(input("Enter the number : "))
    print(predictions[i])
    print(test_labels[i])