import equinox as eqx
from jax import lax
import jax.numpy as jnp
import jax

import FEIN.utils.nn as nn_utils


class ResNet(eqx.Module):
    state_increment: eqx.Module

    def __init__(self, mlp_params: nn_utils.MLPParameters, *, key) -> None:
        self.state_increment = eqx.nn.MLP(
            **mlp_params.__dict__,
            key=key
        )

    @eqx.filter_jit
    def __call__(self, x_0, U):
        def body_fcn(carry, input):
            mlp_input = jnp.concatenate((carry, input))
            x_next = carry + self.state_increment(mlp_input)
            return x_next, x_next

        _, states = lax.scan(body_fcn, x_0, U)
        return jnp.vstack((x_0, states))

