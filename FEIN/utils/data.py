import jax.numpy as jnp
import jax
import yaml
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import Protocol
import pandas as pd
import copy


class JAXMinMaxScalar:
    """ Scikit scalar uses numpy arrays or stores the min and max values
    of the dataset in numpy arrays. In this implementation Jax arratys
    are used to avoid mixing numpy and jax arrays during training
    """
    def __init__(self, feature_range: tuple) -> None:
        self.min, self.max = feature_range

    def fit(self, x):
        self.data_min_ = jnp.min(x, axis=0)
        self.data_max_ = jnp.max(x, axis=0)

    def transform(self, x):
        x_std = jnp.divide((x - self.data_min_), (self.data_max_ - self.data_min_))  
        return x_std * (self.max - self.min) + self.min
    
    def vtransform(self, x):
        return jax.vmap(self.transform)(x)

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

@dataclass
class DLODataset:
    U_encoder: jnp.ndarray = None
    U_dyn: jnp.ndarray = None
    U_decoder: jnp.ndarray = None
    Y: jnp.ndarray = None
    encoder_scalar: JAXMinMaxScalar = None
    dyn_scalar: JAXMinMaxScalar = None
    output_scalar: JAXMinMaxScalar = None


class DLODatasetScalars(Protocol):
    encoder_scalar: JAXMinMaxScalar = None
    dyn_scalar: JAXMinMaxScalar = None
    output_scalar: JAXMinMaxScalar = None


class DLOTrajectory:
    def __init__(
        self, 
        traj_path: str,
        n_traj: int = None,
        idxs: tuple = None
    ) -> None:
        axes = ['x', 'y', 'z']
        self.dt = 0.004
        self.n_traj = n_traj
        self.data = pd.read_csv(traj_path)

        if idxs is not None:
            self.data = self.data.loc[idxs, :]
        
        self.t = jnp.array(self.data[['t']].values)

        self.p_b_cols = [f'p_b_{x}' for x in axes]
        self.phi_b_cols = [f'phi_b_{x}' for x in reversed(axes)]
        self.dp_b_cols = [f'dp_b_{x}' for x in axes]
        self.dphi_b_cols = [f'dphi_b_{x}' for x in reversed(axes)]
        self.ddp_b_cols = [f'ddp_b_{x}' for x in axes]
        self.ddphi_b_cols = [f'ddphi_b_{x}' for x in reversed(axes)]
        self.p_e_cols = [f'p_e_{x}' for x in axes]
        self.dp_e_cols = [f'dp_e_{x}' for x in axes]

        self.p_b = jnp.array(self.data[self.p_b_cols].values)
        self.phi_b = jnp.array(self.data[self.phi_b_cols].values)
        self.dp_b = jnp.array(self.data[self.dp_b_cols].values)
        self.dphi_b = jnp.array(self.data[self.dphi_b_cols].values)
        self.ddp_b = jnp.array(self.data[self.ddp_b_cols].values)
        self.ddphi_b = jnp.array(self.data[self.ddphi_b_cols].values)
        self.p_e = jnp.array(self.data[self.p_e_cols].values)
        self.dp_e = jnp.array(self.data[self.dp_e_cols].values)

    def get_sliced_copy(self, idxs):
        sliced_obj = copy.copy(self)

        # Slice attributes using the provided indices
        for attr_name in dir(self):
            if not attr_name.startswith("__") and not callable(getattr(self, attr_name)):
                attr = getattr(self, attr_name)
                if isinstance(attr, jnp.ndarray):
                    setattr(sliced_obj, attr_name, attr[idxs,:])
        return sliced_obj

    def __len__(self):
        return self.t.shape[0]
    
    def plot(self, variable):
        if isinstance(variable, list):
            vars, lbls = [], []
            for v in variable:
                if hasattr(self, variable):
                    vars.append(getattr(self, v))
                    lbls.append(getattr(self, v + '_cols'))
            plot_trajs([self.t]*len(vars), vars, lbls)
        else:
            if hasattr(self, variable):
                plot_trajs(self.t, getattr(self, variable), getattr(self, variable + '_cols'))


class DLODataLoader:
    def __init__(
        self, 
        U_enc: jnp.ndarray, 
        U_dyn: jnp.ndarray, 
        U_dec: jnp.ndarray,
        Y: jnp.ndarray, 
        key: jax.random.PRNGKey
    ) -> None:

        self.key = key
        self.U_enc = U_enc
        self.U_dec = U_dec
        self.U_dyn = U_dyn
        self.Y = Y
        self.n_rollouts = self.Y.shape[0]
        self.indices = jnp.arange(self.n_rollouts)

    def get_batch(self, batch_size: int):
        self.key, subkey = jax.random.split(self.key)
        batch_idxs = jax.random.choice(subkey, self.indices, shape=(batch_size,), replace=False)
        return self.U_enc[batch_idxs], self.U_dyn[batch_idxs], self.U_dec[batch_idxs], self.Y[batch_idxs] 

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, idx):
        return self.Y[idx], self.U[idx]


def load_vicon_marker_locations(path_to_yaml: str):
    with open(path_to_yaml, 'r') as file:
        marker_data = yaml.safe_load(file)

    marker_dist = dict()
    for keys, values in marker_data.items():
        marker_dist[keys] = np.linalg.norm(values)
    
    return marker_dist


def load_trajs(n_trajs: list, dlo: str):
    if dlo not in ['aluminium-rod', 'pool-noodle']:
        raise ValueError('Please provide a valid DLO name')
    dataset_path = f'dataset/{dlo}/'
    
    trajfile_names = [f'traj{n}.csv' for n in n_trajs]
    trajfile_paths = [dataset_path + t for t in trajfile_names]
    trajs = [DLOTrajectory(t, n) for t, n in zip(trajfile_paths, n_trajs)]
    return trajs


def plot_trajs(T, Y, lbls: list):
    # to treat rollouts and trajectories similarly
    if not isinstance(T, list):
        T, Y = [T], [Y]

    # Sanity check
    n_cols = Y[0].shape[1] // 3

    _, axs = plt.subplots(3, n_cols, sharex=True)
    axs = axs.T.reshape(-1)
    for t, y in zip(T, Y):
        for k, ax in enumerate(axs):
            ax.plot(t, y[:,k])
            ax.set_ylabel(lbls[k])
            ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


