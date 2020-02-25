import unittest
import numpy as np
import time
import torch
from torch import nn
from autoPyTorch.pipeline.base.pipeline import Pipeline
from autoPyTorch.utils.configspace_wrapper import ConfigWrapper
from ConfigSpace.read_and_write import json
from autoPyTorch.pipeline.nodes.loss_module_selector import AutoNetLossModule
from autoPyTorch.pipeline.nodes.train_node import TrainNode, Trainer
from autoPyTorch.pipeline.nodes.create_dataloader import CreateDataLoader
from autoPyTorch.pipeline.nodes.create_dataset_info import CreateDatasetInfo
from autoPyTorch.pipeline.nodes.network_selector import NetworkSelector
from autoPyTorch.pipeline.nodes.optimizer_selector import OptimizerSelector
from autoPyTorch.components.optimizer.optimizer import AdamOptimizer
from autoPyTorch.pipeline.nodes.metric_selector import MetricSelector, no_transform
from autoPyTorch.components.metrics.standard_metrics import accuracy, auc_metric, mae
from autoPyTorch.pipeline.nodes.log_functions_selector import AutoNetLog
from autoPyTorch.components.networks.feature.mlpnet import MlpNet
from autoPyTorch.pipeline.nodes.loss_module_selector import LossModuleSelector
from autoPyTorch.components.preprocessing.loss_weight_strategies import LossWeightStrategyWeighted
import torch.optim as optim
from autoPyTorch.components.training.early_stopping import EarlyStopping
from autoPyTorch.components.training.budget_types import BudgetTypeEpochs
import os


def get_data_loaders(hyperparameter_config, X, y, train_indices, valid_indices):
    dl = hyperparameter_config.get_hyperparameter(CreateDataLoader.get_name() +  ConfigWrapper.delimiter + "batch_size")
    hyperparameter_config = dict()
    hyperparameter_config[dl.name] = 2
    pipeline_config = dict()
    pipeline_config['random_seed'] =  42
    create_dataloader_node = CreateDataLoader()
    result = create_dataloader_node.fit(pipeline_config, hyperparameter_config, X, y, train_indices, valid_indices)
    return result["train_loader"], result["valid_loader"]

def get_loss_function():
    pipeline = Pipeline([
        LossModuleSelector()
    ])
    selector = pipeline[LossModuleSelector.get_name()]
    selector.add_loss_module("L1", nn.L1Loss)
    selector.add_loss_module("cross_entropy", nn.CrossEntropyLoss, LossWeightStrategyWeighted(), True)
    pipeline_config = pipeline.get_pipeline_config(loss_modules=["L1", "cross_entropy"])
    pipeline_hyperparameter_config = pipeline.get_hyperparameter_search_space(**pipeline_config).sample_configuration()
    pipeline_hyperparameter_config["LossModuleSelector:loss_module"] = "L1"
    pipeline.fit_pipeline(hyperparameter_config=pipeline_hyperparameter_config, train_indices=np.array([0, 1, 2]), X=np.random.rand(3,3), Y=np.random.rand(3, 2), pipeline_config=pipeline_config, tmp=None)
    selected_loss = pipeline[selector.get_name()].fit_output['loss_function']
    return selected_loss

def get_metric():
    pipeline = Pipeline([
        MetricSelector()
    ])
    selector = pipeline[MetricSelector.get_name()]
    selector.add_metric("auc", auc_metric)
    selector.add_metric("accuracy", accuracy)
    selector.add_metric("mean", mae)
    pipeline_config = pipeline.get_pipeline_config(optimize_metric="accuracy", additional_metrics=['auc', 'mean'])
    pipeline.fit_pipeline(pipeline_config=pipeline_config)
    selected_optimize_metric = selector.fit_output['optimize_metric']
    selected_additional_metrics = selector.fit_output['additional_metrics']
    return selected_optimize_metric, selected_additional_metrics

def get_network_optimizer(X, y):
    pipeline = Pipeline([
        NetworkSelector(),
        OptimizerSelector()
    ])
    net_selector = pipeline[NetworkSelector.get_name()]
    net_selector.add_network("mlpnet", MlpNet)
    net_selector.add_final_activation('none', nn.Sequential())
    opt_selector = pipeline[OptimizerSelector.get_name()]
    opt_selector.add_optimizer("adam", AdamOptimizer)
    pipeline_config = pipeline.get_pipeline_config()
    pipeline_config["random_seed"] = 42
    hyper_config = pipeline.get_hyperparameter_search_space().sample_configuration()
    pipeline.fit_pipeline(hyperparameter_config=hyper_config, pipeline_config=pipeline_config, 
                            X=X, Y=y, embedding=nn.Sequential())
    sampled_optimizer = opt_selector.fit_output['optimizer']
    selected_network = net_selector.fit_output['network']
    return selected_network, sampled_optimizer


class TestTrainNode(unittest.TestCase):
    def test_train_node(self):
        path_prefix = os.path.abspath(os.path.dirname(__file__))
        with open(os.path.join(path_prefix + '/hyperparameter_config.json'), 'r') as fh:
            json_string = fh.read()     
            full_hyperparameter_config = json.read(json_string)

        len_x = 15
        X = np.random.rand(len_x, 3)
        train_indices = np.random.choice(len_x, size=7, replace=False)
        valid_indices = np.setdiff1d(np.arange(len_x), train_indices, True)
        y = np.random.choice(2, size=len(X)).reshape(len(X), -1)

        pipeline = Pipeline([
            TrainNode()
        ])

        train_loader, valid_loader = get_data_loaders(full_hyperparameter_config, X, y, train_indices, valid_indices)
        train_node = pipeline[TrainNode().get_name()]
        network, optimizer = get_network_optimizer(X, y)
        optimize_metric, additional_metrics = get_metric()
        log_functions = None
        training_techniques = [BudgetTypeEpochs()]
        refit = False
        fit_start_time = time.time()
        budget = 1
        pipeline_config = dict()
        options = train_node.get_pipeline_config_options()


        for option in options:
            pipeline_config[option.name] = option.default


        hyperparameter_config = full_hyperparameter_config.sample_configuration()
        loss_function = get_loss_function()
        pipeline_config['cuda'] = False
        try:
            result = train_node.fit(hyperparameter_config, pipeline_config,
                            train_loader, valid_loader,
                            network, optimizer,
                            optimize_metric, additional_metrics,
                            log_functions,
                            budget,
                            loss_function,
                            training_techniques,
                            fit_start_time,
                            refit)
            self.assertIn('loss', result.keys())
            self.assertIn('info', result.keys())
        except:
            self.assertRaises(ValueError) #catching sklearn value error due to Only one class present in y_true. ROC AUC score is not defined in that case.

        