import pandas as pd
import time
import pickle
import os
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

from demo.model import model
from demo.preprocess import preprocess
from demo.util import utils

Current_directory = os.path.dirname(os.path.abspath(__file__))

ROOT_DIR = (os.path.abspath(os.path.join(Current_directory, os.pardir)))


class Train:

    def __init__(self, train_data_path,
                 vectorizer_file_name='vectorizer-v2.pickel',
                 binarizer_file_name='binarizer-v2.pickel',
                 model_file_name='model-n2-n4-90-v2.h5'):

        self.train_data_path = train_data_path
        self.vectorizer_file = vectorizer_file_name
        self.binarizer_file = binarizer_file_name
        self.model_file = model_file_name

    # Read the data
    def load_train(self):
        df_full = pd.read_csv(self.train_data_path)
        return df_full

    def train(self, train_data, num_epochs=50, batch_size=128):

        start_time = time.time()

        # clean the reports in the data frame
        train_data['PREPROCESSED_REPORT'] = preprocess.clean_text(train_data['PREPROCESSED_REPORT'])

        # only consider rows with report size greater than 6
        train_data = train_data[train_data['PREPROCESSED_REPORT'].map(len) > 6]

        print('Cleaned data shape:',train_data.shape)

        x_full = train_data["PREPROCESSED_REPORT"]
        y_full = train_data['ICD9_CODE'].apply(lambda x: utils.get_list_from_str(x))

        count_vectorizer = TfidfVectorizer(min_df=50, max_df=0.7)
        mb = MultiLabelBinarizer()

        x = count_vectorizer.fit_transform(x_full)
        y = mb.fit_transform(y_full)
        print(x.shape, y.shape)

        vectorizer_path = os.path.join(ROOT_DIR, 'model/trained_models', self.vectorizer_file)
        print(vectorizer_path)

        with open(vectorizer_path, 'wb') as v:
            pickle.dump(count_vectorizer, v)

        binarizer_path = os.path.join(ROOT_DIR,'model/trained_models', self.binarizer_file)
        print(binarizer_path)

        with open(binarizer_path, 'wb') as b:
            pickle.dump(mb, b)

        n_features = x.shape[1]
        labels = mb.classes_
        num_classes = len(labels)

        # write the labels to memory
        np.save(os.path.join(ROOT_DIR,'data','labels.npy'), labels)

        classifier = model.create_model_architecture(n_features, num_classes)

        model_path = os.path.join(ROOT_DIR,'model/trained_models', self.model_file)
        callbacks = [
            ReduceLROnPlateau(),
            EarlyStopping(patience=4),
            ModelCheckpoint(filepath=model_path, save_best_only=True)
        ]
        print('Training start at:', time.time())

        classifier.fit(x, y,
                       batch_size=batch_size,
                       validation_split=0.1,
                       epochs=num_epochs,
                       callbacks=callbacks)

        print("Training Completed in:", time.time()-start_time)

        return vectorizer_path, binarizer_path, model_path



