import yaml
import jax
import os
import pytest


import FEIN.utils.data as data_utils
from FEIN import rnn, resnet, neural_ode
from training import run_experiment
import training.preprocess_data as data_prpr


# rnn_test_configs = [f'tests/experiment_configs/rnn_test_{k}.yml' for k in range(1, 7)]
rnn_test_configs = ['tests/experiment_configs/rnn_test_3.yml',
                    'tests/experiment_configs/rnn_test_6.yml']

resnet_test_configs = ['tests/experiment_configs/resnet_test_1.yml',
                       'tests/experiment_configs/resnet_test_2.yml']

node_test_configs = ['tests/experiment_configs/node_test_1.yml',
                     'tests/experiment_configs/node_test_2.yml']


def get_data_loader(config):
    rollout_length = config['rollout_length']
    data_key = jax.random.PRNGKey(config['data_seed']) 
    
    # Load training and validation data
    val_size = 0.15
    data_key, data_subkey = jax.random.split(data_key)
    train_trajs = data_utils.load_trajs(config['train_trajs'])
    train_data, val_data = data_prpr.construct_train_val_datasets_from_trajs(
        train_trajs, rollout_length, val_size, data_subkey
    )
    train_data_loader = data_utils.DLODataLoader(
        train_data.U_encoder,
        train_data.U_dyn,
        train_data.U_decoder,
        train_data.Y,
        data_key   
    )
    return train_data_loader


def test_rnn_model_creation():
    for  config in rnn_test_configs:
        with open(config, 'r') as file:
            config = yaml.safe_load(file)
        m = run_experiment.get_model(config)
        assert isinstance(m.dynamics, rnn.RNN)


def test_resnet_model_creation():
    for config in resnet_test_configs:
        with open(config, 'r') as file:
            config = yaml.safe_load(file)
        m = run_experiment.get_model(config)
        assert isinstance(m.dynamics, resnet.ResNet)


def test_node_model_creation():
    for  config in node_test_configs:
        with open(config, 'r') as file:
                config = yaml.safe_load(file)
        m = run_experiment.get_model(config)
        assert isinstance(m.dynamics, neural_ode.DiscretizedNODE)


def test_rnn_model_computation():
    for k, config in enumerate(rnn_test_configs):
        with open(config, 'r') as file:
            config = yaml.safe_load(file)
        if k == 0:
            data_loader = get_data_loader(config)
            batch_size = config['batch_size']
            U_enc, U_dyn, U_dec, Y = data_loader.get_batch(batch_size)

        m = run_experiment.get_model(config)
        X, Y = jax.vmap(m)(U_enc[:,0,:], U_dyn, U_dec)
        assert (X.shape[0] == batch_size and 
                X.shape[1] == config['rollout_length']+1)
        

def test_resnet_model_computation():
    for k, config in enumerate(resnet_test_configs):
        with open(config, 'r') as file:
            config = yaml.safe_load(file)
        if k == 0:
            data_loader = get_data_loader(config)
            batch_size = config['batch_size']
            U_enc, U_dyn, U_dec, Y = data_loader.get_batch(batch_size)

        m = run_experiment.get_model(config)
        X, Y = jax.vmap(m)(U_enc[:,0,:], U_dyn, U_dec)
        assert (X.shape[0] == batch_size and 
                X.shape[1] == config['rollout_length']+1)


def test_node_model_computation():
    for k, config in enumerate(node_test_configs):
        with open(config, 'r') as file:
            config = yaml.safe_load(file)
        if k == 0:
            data_loader = get_data_loader(config)
            batch_size = config['batch_size']
            U_enc, U_dyn, U_dec, Y = data_loader.get_batch(batch_size)

        m = run_experiment.get_model(config)
        X, Y = jax.vmap(m)(U_enc[:,0,:], U_dyn, U_dec)
        assert (X.shape[0] == batch_size and 
                X.shape[1] == config['rollout_length']+1)


@pytest.mark.skip(reason="Too slow")
def test_rnn_training():
    for config in rnn_test_configs:
        os.system(f"python training/run_experiment.py --wandb_mode disabled --config {config} --save_model False")


@pytest.mark.skip(reason="Too slow")
def test_resnet_training():
    for config in resnet_test_configs:
        os.system(f"python training/run_experiment.py --wandb_mode disabled --config {config}")


@pytest.mark.skip(reason="Too slow")
def test_node_training():
    for config in node_test_configs:
        os.system(f"python training/run_experiment.py --wandb_mode disabled --config {config}")


if __name__ == "__main__":
    test_rnn_training()