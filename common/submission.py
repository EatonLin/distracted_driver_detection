import pandas as pd


class ModelSubmmission(object):

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
        for index, row in new_df.iterrows():
            test_img_path = self.test_path + row['img']
            print(test_img_path)
            prediction = model.predict(test_img_path)
            for i in range(len(classes)):
                new_df.loc[index, classes[i]] = prediction[i]
        new_df.to_csv(new_csv_name, index=False, sep=',')
        print("{} predict completed." % (model.model_name))

