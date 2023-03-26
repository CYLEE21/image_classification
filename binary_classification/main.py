import os
import numpy as np
import keras
from keras.models import *
from keras.layers import *
import imghdr
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

# load the data
data_dir = 'E:\img_data'
dog_img = os.listdir(os.path.join(data_dir, 'dog'))
cat_img = os.listdir(os.path.join(data_dir, 'cat'))

#organize the image_dataset into classes
#dataset folder -> classes folders
# therefore, we use tf dataset to auto loading images.
data = tf.keras.utils.image_dataset_from_directory(data_dir)
# put into the numpy iterator
data_iterator = data.as_numpy_iterator()
# it will iterate the batch every execution
batch = data_iterator.next()
#should be 2 batch[0] is images batch[1] is labels
len(batch)
#(32, 256, 256, 3) as defult, image
batch[0].shape
#(32,) as defult, labels
batch[1].shape
#plot the image
fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])
plt.show()
#0 is wolf, 1 is wolfdog

#scaled all the data from [0,1]
data = data.map(lambda x,y: (x/255, y))
#check out the max and min values
data.as_numpy_iterator().next()[0].max()
data.as_numpy_iterator().next()[0].min()
#the len of data will be 7 becasue the map() will do the lambda function for all
#7 batches
len(data)

#split the data

train_size = int(round(len(data)*.7))
val_size = int(round(len(data)*.2))+1
test_size = int(round(len(data)*.1))
train_size+val_size+test_size

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

#build the model
model = Sequential()
model.add(Conv2D(16, (3,3), 1, input_shape=(256, 256, 3)))
model.add(BatchNormalization())
model.add(MaxPool2D())
model.add(Conv2D(16, (3,3), 1))
model.add(BatchNormalization())
model.add(MaxPool2D())
model.add(Conv2D(16, (3,3), 1))
model.add(BatchNormalization())
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

hist = model.fit(train, epochs=50, validation_data=val)

img = cv2.imread('C:/Users/Brian/image_classification/binary_classification/wolfdog_3.jpg')
resize = tf.image.resize(img, (256, 256))
plt.imshow(resize.numpy().astype(int))
plt.show()
#[256, 256, 3]
resize.shape
#(1, 256, 256, 3)
np.expand_dims(resize, 0).shape

inputs = np.expand_dims(resize/255, 0)

pred = model.predict(inputs)
pred

#using resNet50 for the saliency map
test_model = tf.keras.applications.resnet50.ResNet50()
resize = tf.image.resize(img, (224, 224))
inputs = np.expand_dims(resize/255, 0)
pred_resnet = test_model.predict(inputs)
pred_resnet
#plot saliency map
max_idx = tf.argmax(pred_resnet)
with tf.GradientTape() as tape:
    images = tf.Variable(inputs, dtype=float)
    tape.watch(images)
    result = test_model(images)
    max_score = result[0, max_idx[0]]
    print(max_score)

grads = tape.gradient(max_score, images)
def normalize_image(img):
    grads_norm = img[:,:,0]+ img[:,:,1]+ img[:,:,2]
    grads_norm = (grads_norm - tf.reduce_min(grads_norm))/ (tf.reduce_max(grads_norm)- tf.reduce_min(grads_norm))
    return grads_norm
plt.imshow(normalize_image(grads[0]), cmap='coolwarm')
plt.show()

plt.imshow(normalize_image(images[0]))
plt.show()
