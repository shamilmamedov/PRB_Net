import jax.random as jrandom
from dataclasses import dataclass
import jax.numpy as jnp


@dataclass
class DynamicsInputNoiseParameters:
    sigma_phi: float = 0.0125
    sigma_dpb: float = 0.0125
    sigma_dphi: float = 0.0125
    sigma_ddpb: float = 0.0125
    sigma_ddphi: float = 0.0125


def generate_gaussian_noise(
        sigma: float,
        n_vars: int,
        batch_size: int,
        rollout_length: int,
        key: jrandom.PRNGKey
):
    return sigma * jrandom.normal(key, shape=(batch_size, rollout_length, n_vars)) 


def generate_dynamics_input_noise(noise_params, batch_size, rollout_length, *, key):
    keys = jrandom.split(key, 5)

    phi = generate_gaussian_noise(noise_params.sigma_phi, 3, batch_size, rollout_length, keys[0])
    sin_phi = jnp.sin(phi)
    cos_phi = jnp.cos(phi)
    dpb = generate_gaussian_noise(noise_params.sigma_dpb, 3, batch_size, rollout_length, keys[1])
    dphi = generate_gaussian_noise(noise_params.sigma_dphi, 3, batch_size, rollout_length, keys[2])
    ddpb = generate_gaussian_noise(noise_params.sigma_ddpb, 3, batch_size, rollout_length, keys[3])
    ddphi = generate_gaussian_noise(noise_params.sigma_ddphi, 3, batch_size, rollout_length, keys[4])

    U_dyn_jitter = jnp.concatenate((sin_phi, cos_phi, dpb, dphi, ddpb, ddphi), axis=2)
    return U_dyn_jitter
