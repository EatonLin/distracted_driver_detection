import matplotlib.pyplot as plt
from keras import Model
from keras.applications.inception_v3 import preprocess_input, InceptionV3
from keras.layers import GlobalAveragePooling2D, Dense
from keras.optimizers import Adagrad
from keras.preprocessing.image import ImageDataGenerator

from models.base_model import BaseModel


class InceptionV3Model(BaseModel):
    def __init__(self, resize_path):
        self.model_name = 'inception_v3'
        self.resize_train_path = resize_path + self.model_name + 'train/'
        self.resize_valid_path = resize_path + self.model_name + 'valid/'
        # self.resize_test_path = resize_path + self.model_name + 'test/'
        self.resize_img_shape = (299, 299)
        # datagen = ImageDataGenerator(preprocessing_function=preprocess_input,  # ((x/255)-0.5)*2  归一化到±1之间
        #                              rotation_range=30,
        #                              width_shift_range=0.2,
        #                              height_shift_range=0.2,
        #                              shear_range=0.2,
        #                              zoom_range=0.2,
        #                              horizontal_flip=True)
        self.datagen = {
            'train': ImageDataGenerator(preprocessing_function=preprocess_input,  # ((x/255)-0.5)*2  归一化到±1之间
                                        rotation_range=30,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True),
            'valid': ImageDataGenerator(preprocessing_function=preprocess_input,  # ((x/255)-0.5)*2  归一化到±1之间
                                        rotation_range=30,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True)
        }
        self.generator = {}

    def build_model(self):
        super().build_model()

    def preprocess(self, orgin_df, valid_size, train_path):
        super()._preprocess(orgin_df, valid_size, train_path, self.resize_train_path, self.resize_valid_path)
        self.generator['train'] = self.datagen['train'].flow_from_directory(directory=self.resize_train_path,
                                                                            target_size=self.resize_img_shape,
                                                                            batch_size=64)
        self.generator['valid'] = self.datagen['valid'].flow_from_directory(directory=self.resize_valid_path,
                                                                            target_size=self.resize_img_shape,
                                                                            batch_size=64)

    def __build_model(self):
        base_model = InceptionV3(weights='imagenet', include_top=False)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)  # GlobalAveragePooling2D 将 MxNxC 的张量转换成 1xC 张量，C是通道数
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(10, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        return model

    def __setup_to_transfer_learning(self, model):  # base_model
        for layer in model.layers:
            layer.trainable = True
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def __setup_to_fine_tune(self, model, base_model):
        GAP_LAYER = 10  # max_pooling_2d_2
        for layer in base_model.layers[:GAP_LAYER + 1]:
            layer.trainable = False
        for layer in base_model.layers[GAP_LAYER + 1:]:
            layer.trainable = True
        model.compile(optimizer=Adagrad(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    def __plot_training(history):
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

    def run(self):
        model = self.__build_model()
        self.__setup_to_transfer_learning(model)

        print("开始迁移学习:\n")

        history_ft = model.fit_generator(
            [],
            steps_per_epoch=22424,
            epochs=50,
            validation_data=self.generator['valid'],
            validation_steps=12,
            class_weight='auto')

        print("开始微调:\n")

        # fine-tuning
        self.__setup_to_finetune(model)

        history_ft = model.fit_generator(
            self.generator['train'],
            steps_per_epoch=22424,
            epochs=50,
            validation_data=self.generator['valid'],
            validation_steps=1,
            class_weight='auto')

        model.save("inceptionv3_25000.model")

        self.__plot_training(history_ft)

    def print(self):
        print(self.model_name)
