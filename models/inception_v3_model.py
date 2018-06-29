import matplotlib.pyplot as plt
import numpy as np
import os
from keras import Model
from keras.applications.inception_v3 import preprocess_input, InceptionV3
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import load_model 
from keras.optimizers import Adagrad
from keras.preprocessing.image import ImageDataGenerator

from models.base_model import BaseModel
from preprocess_image import keras_image_load


class InceptionV3Model(BaseModel):
    def __init__(self, resize_path):
        self.model_name = 'inception_v3'
        self.model_file = self.model_name + '.pb'
        self.resize_train_path = resize_path + self.model_name + '/' + 'train/'
        self.resize_valid_path = resize_path + self.model_name + '/' + 'valid/'
        self.resize_test_path = resize_path + self.model_name + '/' + 'test/'
        self.resize_img_shape = (299, 299)
        self.x_train = None
        self.x_valid = None
        self.y_train = None
        self.y_valid = None
        self.history_tl = None
        self.model = None
        self.generator = ImageDataGenerator(preprocessing_function=preprocess_input,
                                        rotation_range=30,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True)

    def preprocess(self, orgin_df, valid_size, train_path):
        if self.x_train is None:
            feature_train, label_train, feature_valid, label_valid = super()._preprocess(orgin_df, valid_size,
                                                                                         train_path,
                                                                                         self.resize_train_path,
                                                                                         self.resize_valid_path,
                                                                                         self.resize_img_shape)
            self.x_train = feature_train
            self.x_valid = feature_valid
            self.y_train = label_train
            self.y_valid = label_valid
        self.generator.fit(self.x_train)

    def __build_model(self):
        base_model = InceptionV3(weights='imagenet', include_top=False)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(10, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        return model

    def __setup_to_transfer_learning(self, model):
        for layer in model.layers:
            layer.trainable = True
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def __setup_to_fine_tune(self, model, base_model):
        GAP_LAYER = 10
        for layer in base_model.layers[:GAP_LAYER + 1]:
            layer.trainable = False
        for layer in base_model.layers[GAP_LAYER + 1:]:
            layer.trainable = True
        model.compile(optimizer=Adagrad(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    def __plot_training(self, history):
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(acc))

        plt.plot(epochs, acc, 'r.')
        plt.plot(epochs, val_acc, 'r')
        plt.title('Training and validation accuracy')

        plt.figure()
        plt.plot(epochs, loss, 'r.')
        plt.plot(epochs, val_loss, 'r-')
        plt.title('Training and validation loss')
        plt.show()

    def run(self, batch_size, epochs):
        if os.path.exists(self.model_file):
            print('Found model, loading...')
            self.model = load_model(self.model_file)
            print('Loading completed.')
            return
        
        print('No train model, start training.')
        model = self.__build_model()
        self.__setup_to_transfer_learning(model)

        print('Transfer learning...')
        history_tl = model.fit(self.x_train,
                               self.y_train,
                               batch_size=batch_size,
                               epochs=epochs,
                               validation_data=(self.x_valid, self.y_valid),
                               shuffle=True)
        self.__plot_training(history_tl)

        print('Fine-tuning...')
        self.__setup_to_fine_tune(model, model)
    
        history_tl = model.fit(self.x_train,
                               self.y_train,
                               batch_size=batch_size,
                               epochs=epochs,
                               validation_data=(self.x_valid, self.y_valid),
                               shuffle=True)
        self.__plot_training(history_tl)

        print('Model saving...')
        model.save(self.model_file)
        print('Completed')
        self.model = model
        
    def plot_train(self):
        self.__plot_training(self.history_tl)
    
    def predict(self, image_path):
        img_array = keras_image_load(image_path, self.resize_img_shape)
        img_array = np.expand_dims(img_array, axis=0)
        return self.model.predict(img_array)

    def print(self):
        print(self.model_name)
