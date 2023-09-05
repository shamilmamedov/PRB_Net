from dataclasses import dataclass
from typing import Callable
import jax.numpy as jnp
import equinox as eqx
import jax


activations = {
    'tanh': jnp.tanh,
    'relu': jax.nn.relu,
    'relu6': jax.nn.relu6,
    'softplus': jax.nn.softplus
}


@dataclass
class MLPParameters:
    in_size: int = 18
    out_size: int = 12
    width_size: int = 32
    depth: int = 2
    activation: Callable = jnp.tanh


@jax.jit
def mse_loss(Y: jnp.ndarray, Y_pred: jnp.ndarray):
    E = Y - Y_pred
    return jnp.mean(jnp.mean(jnp.sum(E**2, axis=2), axis=1))


@jax.jit
def weighted_mse_loss(Y: jnp.ndarray, Y_pred: jnp.ndarray, w_y: jnp.ndarray, w_t: jnp.ndarray):
    E = Y - Y_pred
    weighted_E = (E*jnp.sqrt(w_y))*jnp.sqrt(w_t)
    return jnp.mean(jnp.mean(jnp.sum(weighted_E**2, axis=2), axis=1))


@jax.jit
def l2_loss(x, alpha):
    return alpha * (x ** 2).mean()


@jax.jit
def l1_loss(x, alpha):
    return alpha * jnp.abs(x).mean()


@jax.jit
def mae_loss(Y, Y_pred):
    E = Y - Y_pred
    return jnp.mean(jnp.mean(jnp.sum(jnp.abs(E), axis=2), axis=1))


@jax.jit
def mean_l2_norm(x):
    def _mean_l2_norm(x):
        se = jnp.sum(x**2, axis=1)
        l2_norm = jnp.sqrt(se)
        return jnp.mean(l2_norm)

    return jnp.mean(
        jax.vmap(_mean_l2_norm)(x)
    )