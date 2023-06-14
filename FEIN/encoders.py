import jax
import jax.numpy as jnp
import equinox as eqx

from FEIN.utils.nn import MLPParameters

class NNEncoder(eqx.Module):
    layernorm: eqx.Module
    h_to_xrfem: eqx.Module

    def __init__(self, mlp_params: MLPParameters, *, key) -> None:
        self.layernorm = eqx.nn.LayerNorm(mlp_params.in_size)
        self.h_to_xrfem = eqx.nn.MLP(**mlp_params.__dict__, key=key)

    @eqx.filter_jit
    def __call__(self, h):
        x_rfem = self.h_to_xrfem(self.layernorm(h))
        return x_rfem


class PIEncoder(eqx.Module):
    layernorm: eqx.Module
    h_to_qfrem: eqx.Module

    def __init__(self, mlp_params: MLPParameters, *, key) -> None:
        self.layernorm = eqx.nn.LayerNorm(mlp_params.in_size)
        self.h_to_qfrem = eqx.nn.MLP(**mlp_params.__dict__, key=key)

    @eqx.filter_jit
    def __call__(self, h):
        pos, vel = jnp.split(h, 2)
        q_rfem = self.h_to_qfrem(self.layernorm(pos))
        dq_rfem = jax.jvp(self.h_to_qfrem, (pos,), (vel,))[1]
        return jnp.concatenate((q_rfem, dq_rfem))