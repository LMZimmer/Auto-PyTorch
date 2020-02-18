import unittest
import numpy as np
import time
import os

from torch import nn
from autoPyTorch.pipeline.base.pipeline import Pipeline
from autoPyTorch.utils.configspace_wrapper import ConfigWrapper
from autoPyTorch.pipeline.nodes.preprocessor_selector import PreprocessorSelector
from sklearn.model_selection import train_test_split
from numpy.testing import assert_array_equal
from autoPyTorch.components.preprocessing.feature_preprocessing import *
from ConfigSpace.read_and_write import json
# from sklearn.decomposition import TruncatedSVD
from scipy.sparse import issparse

class TestPreprocessorSelector(unittest.TestCase):
    def test_preprocessor_selector(self):
        X = np.array([[1, 2, 5], 
                    [1.1, 1.2, 1.3], 
                    [3, 5, 7], 
                    [2, 4, 5], 
                    [4, 5, 6],
                    [-1, -3, -2]])
        train_indices = np.array([0, 3, 4])
        valid_indices = np.array([1, 2, 5])
        y = np.array([1, 0, 0, 1, 0, 1])
        preprocessor_dict = {
            "fast_ica": FastICA, 
            "kernel_pca":KernelPCA, 
            "kitchen_sinks":RandomKitchenSinks,
            "nystroem":Nystroem, 
            "polynomial_features":PolynomialFeatures,
            "power_transformer":PowerTransformer, 
            "truncated_svd":TruncatedSVD
            }
        pipeline = Pipeline([
            PreprocessorSelector()
        ])            
        path_prefix = os.path.abspath(os.path.dirname(__file__))
        with open(os.path.join(path_prefix + '/hyperparameter_config.json'), 'r') as fh:
            json_string = fh.read()     
            full_hyperparameter_config = json.read(json_string)

        for key, value in preprocessor_dict.items():
            selector = pipeline[PreprocessorSelector().get_name()]
            selector.add_preprocessor(key, value)

        hyperparameter_config = full_hyperparameter_config.sample_configuration()

        pipeline_config = dict()
        result = selector.fit(hyperparameter_config, pipeline_config, X, y, train_indices, None)

        self.assertEqual(result['X'].shape[1], 2)
        self.assertEqual(issparse(result['X']), False)

