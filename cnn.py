import tensorflow.keras as keras
import numpy as np
import tensorflow as tf


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

import matplotlib.pyplot as plt

plt.imshow(x_train[0, :, :])
plt.colorbar()

x_train = x_train.reshape((60000, 28, 28, 1))
x_train = x_train.astype('float32') / 255
x_test = x_test.reshape((10000, 28, 28, 1))
x_test = x_test.astype('float32') / 255

# x_train = tf.image.rgb_to_grayscale(x_train)
# x_test = tf.image.rgb_to_grayscale(x_test)

from keras.utils import to_categorical

# присваиваем метки классов
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

from keras import models
from keras import layers

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu',  # (3,3) - фильтр
                  input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),  # фильтр (2,2) для пулинга
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),

    layers.Flatten(),
    layers.Dense(128, 'relu'),
    layers.Dense(128, 'relu'),
    layers.Dense(10, 'softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=64)

# ---------------

test_loss, test_acc = model.evaluate(x_test, y_test)
print('TEST: ', test_acc, end='\n\n')

predictions = model.predict(x_train)

IMAGE_ID = 12
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
print(predictions[IMAGE_ID], end='\n\n')
print(class_names[np.argmax(predictions[IMAGE_ID])])
print(y_train[IMAGE_ID], end='\n\n')
