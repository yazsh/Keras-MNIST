from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

def predictImages(model,path, file_count, name):
    print(name)
    for count in range(0,file_count):
        img_path = path + '/pizza' + str(count) + '.jpeg'
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x)
        # decode the results into a list of tuples (class, description, probability)
        # (one such list for each sample in the batch)
        print('Iter:' ,str(count), decode_predictions(preds, top=2)[0])
    print(name)
    print('\n\n\n')

model = ResNet50(weights='imagenet')

predictImages(model,'/Users/yazen/Desktop/datasets/cheesePizza', 30, 'cheese')
predictImages(model,'/Users/yazen/Desktop/datasets/blackOlivePizza', 33, 'blackOlive')
predictImages(model,'/Users/yazen/Desktop/datasets/pepperoniPizza', 32, 'pepperoni')
