from abc import abstractmethod


class BaseModel(object):
    resize_train_path = ''

    def __init__(self):
        pass

    def _preprocess(self, img_data_df, valid_size, origin_path, resize_train_path, resize_valid_path, resize_shape):
        '''数据预处理

        Args:
            img_data_df: 读取的数据表格
            valid_size: 验证集的比例
            origin_path: 源训练集图片路径
            resize_train_path: 修改训练集图片尺寸后的路径
            resize_valid_path: 修改验证集图片尺寸后的路径
            resize_shape: 所修改图片尺寸

        Returns:

        '''
        # image_resize_and_split(img_data_df,
        #                        valid_size,
        #                        origin_path,
        #                        resize_train_path,
        #                        resize_valid_path,
        #                        resize_shape)

    @abstractmethod
    def preprocess(self, img_data_df, valid_size, origin_path):
        pass

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def run_model(self):
        pass
