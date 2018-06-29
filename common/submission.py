import numpy as np
import pandas as pd


class ModelSubmission(object):

    def __init__(self, csv_path, test_path) -> None:
        self.test_path = test_path
        try:
            self.dataset = pd.read_csv(csv_path)
        except:
            print("Dataset could not be loaded. Is the dataset missing?")


    def model_predict(self, model):
        classes = self.dataset.columns.values[1:]
        new_df = self.dataset.copy(deep=True)
        new_csv_name = model.model_name + '_submission.csv'
        new_df.loc[0, classes[0]] = 2
        for index, row in new_df.iterrows():
            test_img_path = self.test_path + row['img']
            prediction = model.predict(test_img_path)
            for i in range(len(classes)):
                new_df.loc[index, classes[i]] = np.round(prediction[0][i], 3)
        new_df.to_csv(new_csv_name, index=False, sep=',')
        print("%s predict completed." % (model.model_name,))

