import numpy as np
import pickle
from matplotlib import pyplot as plt
import cv2
import os
import random
import tensorflow as tf
from tensorflow import keras

X_pickle_in=open("X.pickle","rb")
X_train=pickle.load(X_pickle_in)
print(len(X_train))
test_pickle_in=open("test.pickle","rb")
test_data=pickle.load(test_pickle_in)
print(len(test_data))

y_pickle_in=open("y.pickle","rb")
y=pickle.load(y_pickle_in)

## normalize
X_train=tf.keras.utils.normalize(X_train,axis=1)
test_data=tf.keras.utils.normalize(test_data,axis=1)
# print(len(np.unique(y)))
yzero=[1 for i in range(len(y)) if y[i]==0 ]
yone=[1 for i in range(len(y)) if y[i]==1 ]
ytwo=[1 for i in range(len(y)) if y[i]==2 ]
print("yzero",len(yzero))
print("yone",len(yone))
print("ytwo",len(ytwo))
# exit()

img_size=224
## Explore Data
print("X shape",X_train.shape)
print("y len",len(y))
plt.figure()
plt.imshow(X_train[0])
plt.colorbar()
plt.grid(False)
plt.show()
# exit()

## Build Model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(img_size, img_size,3)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(3, activation='softmax')])

## Compile Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

## Train the model
model.fit(X_train, y, validation_split=0.33, epochs=20)
model.save("trained.model")
# model.fit(X_train, y, epochs=10)
# predictions = model.predict(test_data)
# np.savetxt('predicts.csv', predictions, delimiter=',', fmt='%d')

# Evaluate Accuracy
# test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

