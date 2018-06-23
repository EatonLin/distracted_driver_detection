from keras import Model
from keras.applications.inception_v3 import preprocess_input
from keras.layers import GlobalAveragePooling2D, Dense
from keras.preprocessing.image import ImageDataGenerator

from models.base_model import BaseModel


class InceptionV3(BaseModel):
    def __init__(self):
        self.model_name = 'inception_v3'
        # self.resize_train_path = resize_path + self.model_name + 'train/'
        # self.resize_valid_path = resize_path + self.model_name + 'test/'
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

    def preprocess(self, train_path, test_path):
        self.generator['train'] = self.datagen['train'].flow_from_directory(directory=train_path,
                                                                            target_size=self.resize_img_shape,
                                                                            batch_size=64)
        self.generator['valid'] = self.datagen['valid'].flow_from_directory(directory=test_path,
                                                                            target_size=self.resize_img_shape,
                                                                            batch_size=64)

    def build_model(self):
        base_model = InceptionV3(weights='imagenet', include_top=False)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)  # GlobalAveragePooling2D 将 MxNxC 的张量转换成 1xC 张量，C是通道数
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(10, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

    def run(self):
        pass

    def print(self):
        print(self.model_name)
