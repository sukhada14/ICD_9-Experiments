import pandas as pd
import pickle
import os
import numpy as np
import json

from keras import models
from sklearn.metrics import precision_recall_fscore_support

from demo.preprocess import preprocess
from demo.util import utils

Current_directory = os.path.dirname(os.path.abspath(__file__))

ROOT_DIR = (os.path.abspath(os.path.join(Current_directory, os.pardir)))

# load saved labels
labels = np.load(os.path.join(ROOT_DIR, 'data', 'labels.npy'))
base_probability_value = 0.4

# load icd code descriptions
icd_path = os.path.join(ROOT_DIR, 'data/ref_data', 'description_dictionary.json')
with open(icd_path,'rb') as f:
    icd_description = json.load(f)


class Test:

    def __init__(self, vectorizer_path, binarizer_path, model_path, test_data_path=None):

        self.test_data_path = test_data_path
        self.vectorizer_path = vectorizer_path
        self.binarizer_path = binarizer_path
        self.model_path = model_path

    def load_test(self):

        if self.test_data_path is not None:
            df_full = pd.read_csv(self.test_data_path)
            return df_full
        else:
            raise Exception("Test data path not given.")

    def load_vectorizer_binarizer_model(self):

        with open(self.vectorizer_path, 'rb') as f:
            count_vectorizer = pickle.load(f)

        with open(self.binarizer_path, 'rb') as f:
            binarizer = pickle.load(f)

        model = models.load_model(self.model_path)

        return count_vectorizer, binarizer, model

    def measure(self):

        # load vec, bin and model for prediction
        vectorizer, binarizer, model = self.load_vectorizer_binarizer_model()

        # load test data
        test_data = self.load_test()

        if test_data is not None:

            # clean the reports in the data frame
            test_data['PREPROCESSED_REPORT'] = preprocess.clean_text(test_data['PREPROCESSED_REPORT'])

            # only consider rows with report size greater than 6
            test_data = test_data[test_data['PREPROCESSED_REPORT'].map(len) > 6]

            x_full = test_data["PREPROCESSED_REPORT"]
            y_full = test_data['ICD9_CODE'].apply(lambda x: utils.get_list_from_str(x))

            x = vectorizer.transform(x_full)
            y = binarizer.transform(y_full)

            y_pred = model.predict(x)

            y_pred[y_pred > base_probability_value] = 1
            y_pred[y_pred <= base_probability_value] = 0

            report = precision_recall_fscore_support(y, y_pred, average='micro')

            print("Precision:", report[0])
            print("Recall:", report[1])
            print("f1-score:", report[2])

        return report

    def predict(self, examples, vectorizer, model):

        examples = preprocess.clean_text(examples)

        predicted_result = {}

        for ind, example in enumerate(examples):

            print('Example:', ind)
            
            predicted_labels = []

            x = vectorizer.transform([example])

            y = model.predict(x)[0]

            y[y > base_probability_value] = 1
            y[y <= base_probability_value] = 0

            predicted_indices = np.argwhere(y == np.amax(y)).flatten().tolist()

            for i in predicted_indices:

                predicted_labels[labels[i]] = icd_description[labels[i]]

            predicted_result[ind] = predicted_labels

        return predicted_result
                














