import jax.random as jrandom
import matplotlib.pyplot as plt
from dataclasses import dataclass
import jax.numpy as jnp

import training.preprocess_data as data_pp
from training import jittering




def main(
        rollout_length: int = 250
):
    sigma = 0.025 # phi = 0.0125, dphi = 0.025 ddphi = 0.05 sigma_dpb = 0.025 sigma_ddpb
    key = jrandom.PRNGKey(1701)
    key, subkey = jrandom.split(key)
    
    n_vars = 3
    batch_size = 1
    jitter = jittering.generate_gaussian_noise(sigma, n_vars, batch_size, rollout_length, key=subkey).squeeze()

    n_trajs = [19]
    trajs = data_pp.load_trajs(n_trajs)
    train_data, val_data = data_pp.construct_train_val_datasets_from_trajs(
        trajs, rollout_length, 0.15, key
    )
    trajs = trajs[0]

    nw = 20
    idx_i = nw*rollout_length
    idx_f = (nw+1)*rollout_length

    _, axs = plt.subplots(3,1,sharex=True)
    axs.reshape(-1)
    for k, ax in enumerate(axs):
        ax.plot(trajs.dp_b[idx_i:idx_f,k])
        ax.plot(trajs.dp_b[idx_i:idx_f,k] + jitter[:,k])
        # ax.axhline(y=sigma_phi, color='r', linestyle='--')
        # ax.axhline(y=-sigma_phi, color='r', linestyle='--')
    plt.tight_layout()
    plt.show()


def test_generate_dynamics_input_noise():
    noise_params = jittering.DynamicsInputNoiseParameters()

    key = jrandom.PRNGKey(1701)
    
    batch_size = 64
    rollout_length = 250

    U_dyn_jitter = jittering.generate_dynamics_input_noise(noise_params, batch_size, rollout_length, key=key)

    assert U_dyn_jitter.shape[0] == batch_size
    assert U_dyn_jitter.shape[1] == rollout_length
    assert U_dyn_jitter.shape[2] == 18


if __name__ == '__main__':
    test_generate_dynamics_input_noise()
    main()