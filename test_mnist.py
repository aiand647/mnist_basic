'''
encoding : utf-8
license MIT
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

#load data
(x_train,y_train),(x_test,y_test)=mnist.load_data(path='mnist.npz')

#normalization
x_train=tf.keras.utils.normalize(x_train, axis=1)
x_test=tf.keras.utils.normalize(x_test, axis=1)

#loading saved model
model = tf.keras.models.load_model('{path/to/saved/name.h5/file}')

#details
model.summary()

loss,acc=model.evaluate(x=x_test,y=y_test,verbose=2)
print('Restored model, accuracy: {fcc}%'.format(fcc=round(100*acc,2)))
#test by selecting a image by choosing a number from 1-1000
n=1
while n!=0:
    n=int(input("Enter b/w 1-1000: "))

    predictions=model.predict(x_test)

    print(np.argmax(predictions[n]))

    plt.imshow(x_test[n], cmap="gray")
    plt.show()