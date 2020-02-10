import unittest
import numpy as np
import time

from torch import nn
from autoPyTorch.pipeline.nodes.embedding_selector import EmbeddingSelector
from autoPyTorch.pipeline.base.pipeline import Pipeline
from autoPyTorch.utils.configspace_wrapper import ConfigWrapper
from autoPyTorch.pipeline.nodes.create_dataset_info import CreateDatasetInfo
from autoPyTorch.components.networks.feature import LearnedEntityEmbedding
import json


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

        # y = np.array([1, 0, 0, 1, 0, 1])

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        selector = pipeline[EmbeddingSelector().get_name()]
        selector.add_embedding_module('learned', LearnedEntityEmbedding)
        options = selector.get_pipeline_config_options()
        pipeline_config = dict()
        for option in options:
            pipeline_config[option.name] = option.default 

        hyperparameter_config = selector.get_hyperparameter_search_space(pipeline_config).sample_configuration()
        hyperparameter_config['embeddings'] = 'learned'
        
        result = selector.fit(hyperparameter_config, pipeline_config, X, False)
        self.assertEqual(type(result['embedding']), type(nn.Sequential()))

