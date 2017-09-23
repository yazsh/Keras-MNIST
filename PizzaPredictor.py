import keras as ks
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import Model
import ssl
import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

epochs = 10
steps_per_epoch = 120
batch_size = 10

preTrained_model = VGG16(include_top=False,weights='imagenet')
datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

generator = datagen.flow_from_directory(
        '/Users/Yazen Shunnar/Desktop/Pizza1/',
        target_size=(244, 244),
        batch_size=batch_size,
        class_mode='binary',  # this means our generator will only yield batches of data, no labels
        shuffle=False)


topModel = preTrained_model.output
topModel = ks.layers.GlobalAveragePooling2D()(topModel)
topModel = ks.layers.Dense(1, activation='sigmoid')(topModel)
# topModel = ks.layers.Dense(50,activation='relu')(topModel)
# topModel = ks.layers.Dense(50, activation='relu')(topModel)
# topModel = ks.layers.Dropout(0.5)(topModel)
# topModel = ks.layers.Dense(50, activation='relu')(topModel)
model = Model(inputs=preTrained_model.input, outputs=topModel)


for layer in model.layers[:25]:
    layer.trainable = False


model.compile(loss='binary_crossentropy',
              optimizer=ks.optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

model.fit_generator(generator,steps_per_epoch=steps_per_epoch,epochs=epochs)


def predictImages(model, path, file_count, name):
    print('\n')
    sum1 = 0
    for count in range(0,file_count):
        img_path = path + '/pizza' + str(count) + '.jpeg'
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x)
        sum1 += np.round(preds)
        print(preds)
        # decode the results into a list of tuples (class, description, probability)
        # (one such list for each sample in the batch)
        # print('Iter:' ,str(count), decode_predictions(model.predict(x), top=1)[0])
    if(name == 'cheese'):


        print(name + ":  " + str(sum1/file_count))
    else:
        print(name + ":  " + str(1 - sum1/file_count))

    print('\n\n')

# model.fit_generator(generate_arrays_from_file(generate_arrays_from_file('/Users/yazen/Desktop/datasets/cheesePizza'), steps_per_epoch=10000, epochs=10)

predictImages(model,'/Users/Yazen Shunnar/Desktop/Pizza1/cheesePizza', 30, 'cheese')
predictImages(model,'/Users/Yazen Shunnar/Desktop/Pizza1/blackOlivePizza', 33, 'blackOlive')
# predictImages(newModel,'/Users/yazen/Desktop/datasets/pepperoniPizza', 32, 'pepperoni')
print(generator.class_indices)
