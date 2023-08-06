import jax.numpy as jnp
import equinox as eqx

from FEIN.utils.ode import IntegratorSetting, simulate_ode
from FEIN.utils.nn import MLPParameters


class Integrator(eqx.Module):
    setting: IntegratorSetting

    def __init__(self, setting: IntegratorSetting) -> None:
        self.setting = setting

    @eqx.filter_jit
    def __call__(self, ode: eqx.Module, x_0, U):
        return simulate_ode(ode, x_0, U, self.setting)


class NODE(eqx.Module):
    """ Pure neural ode where the whole vector field is modeled as NN"""
    vector_field: eqx.Module
    layernorm: eqx.Module

    def __init__(self, mlp_params: MLPParameters, *, key) -> None:
        self.vector_field = eqx.nn.MLP(
            **mlp_params.__dict__,
            key=key
        )
        self.layernorm = eqx.nn.LayerNorm(mlp_params.in_size)

    @eqx.filter_jit        
    def __call__(self, x, u):
        xu = jnp.concatenate((x,u)) 
        return self.vector_field(self.layernorm(xu))


class SONODE(eqx.Module):
    """Second oderder neural ode: it means that the dynamics is
    ddq = f(q, dq, u) -> x = [q, dq]
    dx = [dq, f(q, dq, u)]
    """
    vector_field: eqx.Module
    layernorm: eqx.Module

    def __init__(self, mlp_params: MLPParameters, *, key) -> None:
        self.vector_field = eqx.nn.MLP(
            **mlp_params.__dict__,
            key=key
        )
        self.layernorm = eqx.nn.LayerNorm(mlp_params.in_size)

    def __call__(self, x, u):
        q, dq = jnp.split(x, 2)
        

        xu = jnp.concatenate((x,u))
        ddq = self.vector_field(self.layernorm(xu))
        return jnp.concatenate((dq, ddq))


class DiscretizedNODE(eqx.Module):
    node: eqx.Module
    integrator: eqx.Module

    def __init__(self, node, integrator) -> None:
        self.node = node
        self.integrator = integrator

    @eqx.filter_jit  
    def __call__(self, x_0, U):
        return self.integrator(self.node, x_0, U)
