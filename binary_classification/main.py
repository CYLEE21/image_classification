import os
import numpy as np
import keras
from keras.models import *
from keras.layers import *
import imghdr
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
data_dir = 'E:\img_data'

image_exts = ['jpeg', 'jpg', 'bmp', 'png']
dog_img = os.listdir(os.path.join(data_dir, 'dog'))
cat_img = os.listdir(os.path.join(data_dir, 'cat'))

#organize the image_dataset into classes
#dataset folder -> classes folders
data = tf.keras.utils.image_dataset_from_directory('E:\img_data')

data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()
#should be 2
len(batch)
#(32, 256, 256, 3) as defult, image
batch[0].shape
#(32,) as defult, labels
batch[1].shape

fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])
plt.show()
#1 is wolfdog, 0 is wolf

#scaled all the data from [0,1]
data = data.map(lambda x,y: (x/255, y))
#check out the max and min values
data.as_numpy_iterator().next()[0].max()
data.as_numpy_iterator().next()[0].min()
#the len of data will be 7 becasue the map() will do the lambda function for all
#7 batches
len(data)

#split the data
train_size = int(len(data)*.7)
val_size = int(len(data)*.2)+1
test_size = int(len(data)*.1)+1
train_size+val_size+test_size

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)


model = Sequential()
model.add(Conv2D(16, (3,3), 1, input_shape=(256, 256, 3)))
model.add(MaxPool2D())
model.add(Conv2D(16, (3,3), 1))
model.add(MaxPool2D())
model.add(Conv2D(16, (3,3), 1))
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

hist = model.fit(train, epochs=50, validation_data=val)

test_model = tf.keras.applications.resnet50.ResNet50()


img = cv2.imread('labura.jpg')
plt.imshow(img)
plt.show()

resize = tf.image.resize(img, (224, 224))
plt.imshow(resize.numpy().astype(int))
plt.show()
#[256, 256, 3]
resize.shape
#(1, 256, 256, 3)
np.expand_dims(resize, 0).shape

inputs = np.expand_dims(resize/255, 0)

pred = model.predict(inputs)
pred

pred_resnet = test_model.predict(inputs)
pred_resnet

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
plt.imshow(normalize_image(grads[0]), cmap='ocean')
plt.show()

plt.imshow(normalize_image(images[0]))
plt.show()






images = tf.Variable(inputs, dtype=float)

with tf.GradientTape() as tape:
    images = tape.watch(images)
    pred = model(images, training=False)
    class_idxs_sorted = np.argsort(pred.numpy().flatten())[::-1]
    loss = pred[0][class_idxs_sorted[0]]
    
grads = tape.gradient(loss, images)
dgrad_abs = tf.math.abs(grads)
dgrad_max_ = np.max(dgrad_abs, axis=3)[0]
arr_min, arr_max  = np.min(dgrad_max_), np.max(dgrad_max_)
grad_eval = (dgrad_max_ - arr_min) / (arr_max - arr_min + 1e-18)

fig, axes = plt.subplots(1,2,figsize=(14,5))
axes[0].imshow(resize.numpy().astype(int))
i = axes[1].imshow(grad_eval,cmap="jet",alpha=0.8)
fig.colorbar(i)
plt.show()




pred = model.predict(test)

y_pred_list = []
y_true_list = []

for images, labels in test.take(1):
    numpy_images = images.numpy()
    numpy_labels = labels.numpy()
len(numpy_labels)
len(pred)
for pred, true in zip(pred, numpy_labels):
  y_pred_list.append(np.argmax(pred))
  y_true_list.append(np.argmax(true))

from sklearn.metrics import confusion_matrix as cm
print(cm(y_true_list, y_pred_list))