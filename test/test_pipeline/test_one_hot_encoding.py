import unittest
import numpy as np
import time

from torch import nn
from autoPyTorch.pipeline.nodes.one_hot_encoding import OneHotEncoding
from autoPyTorch.pipeline.base.pipeline import Pipeline
from autoPyTorch.utils.configspace_wrapper import ConfigWrapper
from autoPyTorch.pipeline.nodes.create_dataset_info import CreateDatasetInfo, DataSetInfo
from sklearn.model_selection import train_test_split
from numpy.testing import assert_array_equal
import scipy 
import json

    
class TestOneHotEncoding(unittest.TestCase):

    def test_one_hot_encoding(self):
        # test for X categorical
        X = np.array([[1, 2, 'male'], 
                    [1.1, 1.2, 'female'], 
                    [3, 5, 'unknown'], 
                    [2, 4, 'female'], 
                    [4, 5, 'male'],
                    [-1, -3, 'female']])

        y = np.array([1, 0, 0, 1, 0, 1])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        pipeline = Pipeline([
            OneHotEncoding()
        ])

        pipeline_config = dict()
        pipeline_config['categorical_features'] = [False, False, True]
        encoder = pipeline[OneHotEncoding().get_name()]
        # create_info = CreateDatasetInfo()
        # result = create_info.predict(pipeline_config, X_train, y_train, X_test, y_test)
        # info = resut['dataset_info']
        info = DataSetInfo()
        info.categorical_features = pipeline_config['categorical_features']
        info.is_sparse = scipy.sparse.issparse(X_train)
        info.x_shape = X_train.shape
        info.y_shape = y_train.shape
        info.x_min_value = X_train[:, :-1].astype(dtype=np.float32).min()
        info.x_max_value = X_train[:, :-1].astype(dtype=np.float32).max()
        result = encoder.fit(pipeline_config, X, y, info)
        _, x_encoder, _, y_encoder, info = result.values()
        X_transformed, x_encoder = encoder.predict(pipeline_config, X, x_encoder).values()
        categories = x_encoder.categories_[0].tolist()
        expected_categories = ['female', 'male', 'unknown']
        self.assertCountEqual(categories, expected_categories)
    
    def test_one_hot_encoding_y_categorical(self):
        # test for X and y categorical
        X = np.array([[1, 2, 'male'], 
                    [1.1, 1.2, 'female'], 
                    [3, 5, 'unknown'], 
                    [2, 4, 'female'], 
                    [4, 5, 'male'],
                    [-1, -3, 'female']])

        y = np.array(['one', 'zero', 'zero', 'one', 'zero', 'one'])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        pipeline = Pipeline([
            OneHotEncoding()
        ])

        pipeline_config = dict()
        pipeline_config['categorical_features'] = [False, False, True]
        encoder = pipeline[OneHotEncoding().get_name()]
        # create_info = CreateDatasetInfo()
        # result = create_info.predict(pipeline_config, X_train, y_train, X_test, y_test)
        # info = resut['dataset_info']
        info = DataSetInfo()
        info.categorical_features = pipeline_config['categorical_features']
        info.is_sparse = scipy.sparse.issparse(X_train)
        info.x_shape = X_train.shape
        info.y_shape = y_train.shape
        info.x_min_value = X_train[:, :-1].astype(dtype=np.float32).min()
        info.x_max_value = X_train[:, :-1].astype(dtype=np.float32).max()
        result = encoder.fit(pipeline_config, X, y, info)
        _, x_encoder, _, y_encoder, info = result.values()
        X_transformed, x_encoder = encoder.predict(pipeline_config, X, x_encoder).values()
        categories = x_encoder.categories_[0].tolist()
        expected_categories = ['female', 'male', 'unknown']
        self.assertCountEqual(categories, expected_categories)

