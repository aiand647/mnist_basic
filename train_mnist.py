'''
encoding : utf-8
licence MIT
'''
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

#load the file in as (x_train,y_train),(x_test,y_test)
(x_train,y_train),(x_test,y_test)=mnist.load_data(path='mnist.npz')

#normalisation of pixel to 0-1 range
x_train=tf.keras.utils.normalize(x_train, axis=1)
x_test=tf.keras.utils.normalize(x_test, axis=1)

#Sequential structure
model=tf.keras.models.Sequential()
#convertion of 2D image matrix into 128 cell vector
model.add(tf.keras.layers.Flatten())

#input layer
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
#hidden layer
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
#output layer
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))

#optimizer selection and loss calculation method 
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])
#training phase
model.fit(x=x_train,y=y_train,batch_size=64,epochs=10)

#saving trained weights as .h5 file
model.save('{path/of/the/saved/name.h5/file}')
#print details
test_loss, test_acc=model.evaluate(x=x_test,y=y_test)
print('\nTest accuracy:',test_acc)


