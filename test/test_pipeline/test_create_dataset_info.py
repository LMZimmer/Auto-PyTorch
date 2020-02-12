import unittest
import numpy as np
import time

from torch import nn
from autoPyTorch.pipeline.base.pipeline import Pipeline
from autoPyTorch.utils.configspace_wrapper import ConfigWrapper
from autoPyTorch.pipeline.nodes.create_dataset_info import CreateDatasetInfo, DataSetInfo
from sklearn.model_selection import train_test_split
from numpy.testing import assert_array_equal

import json

    
class TestCreateDatasetInfo(unittest.TestCase):

    def test_create_dataset_info(self):
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
            CreateDatasetInfo()
        ])

        pipeline_config = dict()
        pipeline_config['categorical_features'] = [False, False, True]
        pipeline_config['dataset_name'] = 'CustomDataset'
        creater = pipeline[CreateDatasetInfo().get_name()]
        with self.assertRaises(TypeError) as tm:
            print(tm.exception)
            result = creater.predict(pipeline_config, X_train, y_train, X_test, y_test)
            info = result['dataset_info']

            self.assertEqual(info.name, pipeline_config['dataset_name'])
            self.assertEqual(info.x_shape, X_train.shape)
            self.assertEqual(info.y_shape, y_train.shape)

            non_category = [not a for a in pipeline_config['categorical_features']]
            X_non_categorical = np.apply_along_axis(lambda x: x[non_category], 1, X_train).astype(dtype=np.float32)
            self.assertEqual(info.x_min_value, X_non_categorical.min())
            self.assertEqual(info.x_max_value, X_non_categorical.max())
