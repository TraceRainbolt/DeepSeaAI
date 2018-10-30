import os
import sys
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import InceptionV3
from keras import optimizers, models, layers
import sys
from matplotlib import pyplot as plt

import tensorflow as tf
import keras.backend.tensorflow_backend as ktf

IMG_WIDTH = 150
IMG_HEIGHT = IMG_WIDTH

BATCH_SIZE = 20
EPOCHS = 4

try:
    VISUAL_MODE = sys.argv[1] == '-v'
except:
    VISUAL_MODE = False

MODEL_LOAD = 'sea_classifier_inception.h5'


def get_session():
    gpu_options = tf.GPUOptions(allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def make_model(): 
    model = models.Sequential()

    cnv_layers =  InceptionV3(weights='imagenet', include_top=False,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    model = models.Sequential()
    model.add(cnv_layers)
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(22, activation='sigmoid'))
    
    model.compile(loss='categorical_crossentropy',
     optimizer=optimizers.RMSprop(lr=2e-5),
     metrics=['acc'])

    return model

  
def make_generators(test_size=0.2):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.25)  

    train_generator = datagen.flow_from_directory(
        "cropSeaCreatures",              # Directory containing train data
        target_size=(IMG_HEIGHT, IMG_WIDTH), 
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset="training")       
    
    vld_generator = datagen.flow_from_directory(
        "cropSeaCreatures",              # Validation data directory
        target_size=(IMG_HEIGHT, IMG_WIDTH), 
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset="validation")

    return (train_generator, vld_generator)
    
def main():
    ktf.set_session(get_session())
    train_generator, vld_generator = make_generators()
    if VISUAL_MODE:
        model = models.load_model(MODEL_LOAD)
        visualize(model, train_generator)
    else:
        model = make_model()
        print(model.summary())
        hst = model.fit_generator(
            train_generator,
            steps_per_epoch=100,
            epochs=EPOCHS,
            validation_data=vld_generator,
            validation_steps=20).history


        model.save("sea_classifier_inception.h5")

        for acc, loss, val_acc, val_loss in zip(hst['acc'], hst['loss'],
         hst['val_acc'], hst['val_loss']): 
            print("%.5f / %.5f  %.5f / %.5f" % (acc, loss, val_acc, val_loss))


def visualize(nn, generator):
    print(nn.summary())
    if MODEL_LOAD == 'sea_classifier_inception.h5':
        output = [nn.layers[0].get_output_at(1)] + [layer.output for layer in nn.layers[1:]]
    else:
        output = [layer.output for layer in nn.layers]
    analysisNN = models.Model(input=nn.input, output=output)

    train_images, _ = next(generator)
    train_images = train_images.astype('float32') / 255
    plt.imshow(train_images[idx, :, :, 0])
    plt.show()
   
    images_per_row = 8
    for layer in analysisNN.predict(train_images[0:1])[:3]:
        try:
            width = layer.shape[2]
            height = layer.shape[1]
            num_chls = layer.shape[3]
        except:
            print('Done with visualization.')
            return
        num_rows = num_chls // images_per_row
        display = np.zeros((height * num_rows, width * images_per_row))
      
        for row in range(num_rows):
            for col in range(images_per_row):
                image = layer[0, :, :, row*images_per_row + col]
                image -= image.mean()
                image /= image.std()

                image = np.clip(image, 0, 255).astype('uint8')
                display[row*height:(row+1)*height, col*width:(col+1)*width] = image
                 
        plt.imshow(display)
        plt.show()

main()