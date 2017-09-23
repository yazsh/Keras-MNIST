import keras as ks
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import Model
import ssl
import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import *

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

epochs = 10
steps_per_epoch = 1000
batch_size = 100

model = VGG16(weights='imagenet', include_top=False)
train_dir = '/Users/Yazen Shunnar/Desktop/Pizza1/'

datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

generator = datagen.flow_from_directory(
        train_dir,
        target_size=(244, 244),
        batch_size=batch_size,
        class_mode='binary',  # this means our generator will only yield batches of data, no labels
        shuffle=False)


topModel = Sequential()
topModel.add(GlobalAveragePooling2D(input_shape=model.output_shape[1:]))
topModel.add(ks.layers.Dense(1, activation='sigmoid'))
model = Model(inputs = model.input, outputs=topModel(model.outputs))

for layer in model.layers[:25]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='binary_crossentropy',
              optimizer=ks.optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary')

model.fit_generator(
    train_generator,
    samples_per_epoch=steps_per_epoch,
    epochs=epochs,
    )