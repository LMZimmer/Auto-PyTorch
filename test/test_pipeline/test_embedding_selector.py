import unittest
import numpy as np
import time

from torch import nn
from autoPyTorch.pipeline.nodes.embedding_selector import EmbeddingSelector
from autoPyTorch.pipeline.base.pipeline import Pipeline
from autoPyTorch.utils.configspace_wrapper import ConfigWrapper
from autoPyTorch.pipeline.nodes.create_dataset_info import CreateDatasetInfo, DataSetInfo
from autoPyTorch.pipeline.nodes.one_hot_encoding import OneHotEncoding
from autoPyTorch.components.networks.feature import LearnedEntityEmbedding, NoEmbedding
import json
from sklearn.model_selection import train_test_split
import scipy 
def get_one_hot(X):
    y = np.array([1, 0, 0, 1, 0, 1])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    pipeline = Pipeline([
        OneHotEncoding()
    ])
    pipeline_config = dict()
    pipeline_config['categorical_features'] = [False, False, True]
    encoder = pipeline[OneHotEncoding().get_name()]
    info = DataSetInfo()
    info.categorical_features = pipeline_config['categorical_features']
    info.is_sparse = scipy.sparse.issparse(X_train)
    info.x_shape = X_train.shape
    info.y_shape = y_train.shape
    info.x_min_value = X_train[:, :-1].astype(dtype=np.float32).min()
    info.x_max_value = X_train[:, :-1].astype(dtype=np.float32).max()
    result = encoder.fit(pipeline_config, X, y, info)
    _, x_encoder, _, y_encoder, info = result.values()
    return x_encoder

class TestEmbeddingSelector(unittest.TestCase):

    def test_embedding_selector(self):

        pipeline = Pipeline([
            CreateDatasetInfo(), 
            EmbeddingSelector()
            ])

        X = np.array([[1, 2, 5], 
                    [1.1, 1.2, 1.3], 
                    [3, 5, 7], 
                    [2, 4, 5], 
                    [4, 5, 6],
                    [-1, -3, -2]])

        selector = pipeline[EmbeddingSelector().get_name()]
        selector.add_embedding_module('learned', LearnedEntityEmbedding)
        options = selector.get_pipeline_config_options()
        pipeline_config = dict()
        for option in options:
            pipeline_config[option.name] = option.default 
        hyperparameter_config = selector.get_hyperparameter_search_space(pipeline_config).sample_configuration()
        result = selector.fit(hyperparameter_config, pipeline_config, X, False)
        self.assertEqual(type(result['embedding']), type(nn.Sequential()))

    def test_embedding_selector_categorical(self):
        pipeline = Pipeline([
            CreateDatasetInfo(), 
            EmbeddingSelector()
            ])
        X = np.array([[1, 2, 'male'], 
                    [1.1, 1.2, 'female'], 
                    [3, 5, 'unknown'], 
                    [2, 4, 'female'], 
                    [4, 5, 'male'],
                    [-1, -3, 'female']])

        selector = pipeline[EmbeddingSelector().get_name()]
        selector.add_embedding_module('learned', LearnedEntityEmbedding)
        options = selector.get_pipeline_config_options()
        pipeline_config = dict()
        for option in options:
            pipeline_config[option.name] = option.default 

        hyperparameter_config = selector.get_hyperparameter_search_space(pipeline_config).sample_configuration().get_dictionary()

        hyperparameter_config = ConfigWrapper(selector.get_name(), hyperparameter_config)

        one_hot_encoder = get_one_hot(X)

        with self.assertRaises(AttributeError) as e:
            result = selector.fit(hyperparameter_config, pipeline_config, X, one_hot_encoder)
            assert issubclass (type(result['embedding']), nn.Module)