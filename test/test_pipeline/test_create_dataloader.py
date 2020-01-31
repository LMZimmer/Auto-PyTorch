import unittest
import numpy as np
import time

from autoPyTorch.pipeline.nodes.create_dataloader import CreateDataLoader
from autoPyTorch.pipeline.base.pipeline import Pipeline
from autoPyTorch.utils.configspace_wrapper import ConfigWrapper
from ConfigSpace.read_and_write import json
from autoPyTorch.utils.config_space_hyperparameter import add_hyperparameter
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH


class TestCreateDataLoader(unittest.TestCase):

    def test_create_dataloader(self):
        X = np.array([[1, 2, 5], 
                    [1.1, 1.2, 1.3], 
                    [3, 5, 7], 
                    [2, 4, 5], 
                    [4, 5, 6],
                    [-1, -3, -2]])
        train_indices = np.array([0, 3, 4])
        valid_indices = np.array([1, 2, 5])
        y = np.array([1, 0, 0, 1, 0, 1])
        with open('hyperparameter_config.json', 'r') as fh:
            json_string = fh.read()     
            full_hyperparameter_config = json.read(json_string)

        hyperparameter_config = dict()
        dl = full_hyperparameter_config.get_hyperparameter(CreateDataLoader.get_name() +  ConfigWrapper.delimiter + "batch_size")
        hyperparameter_config[dl.name] = 2
        pipeline_config = dict()
        pipeline_config['random_seed'] =  42
        create_dataloader_node = CreateDataLoader()
        result = create_dataloader_node.fit(pipeline_config, hyperparameter_config, X, y, train_indices, valid_indices)
        train_loader, valid_loader, batch_size = result["train_loader"], result["valid_loader"], result["batch_size"]

        assert batch_size == 2
        for step, (input, target) in enumerate(train_loader):
            assert len(input) == len(target) == batch_size
        
        for step, (input, target) in enumerate(valid_loader):
            assert len(input) == len(target) == batch_size
        