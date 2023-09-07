import jax.numpy as jnp

from FEIN.rfem_kinematics import rfem, models
import FEIN.utils.kinematics as jutils

# unit test for create_custom_model that
# first loads aluminium rod params using load_aluminium_rod_params
# then defines the marker position, and finally creates the model
def test_create_custom_model():
    n_seg = 5
    p_markers = [jnp.array([[1.92, 0., 0.]]).T]

    dlo_params = models.load_aluminium_rod_params()
    rfem_params = models.RFEMParameters(
        n_seg, dlo_params, jutils.JointType.FREE, p_markers
    )
    model = models.create_rfem_custom_model(rfem_params)
    assert model.n_frames == 1
    assert model.n_bodies == n_seg + 1
    assert model.n_joints == n_seg + 1


# unit test for create_rfem_pinocchio_model 
# that first loads aluminium rod params using load_aluminium_rod_params
# then defines the marker position, and finally creates the model
def test_create_rfem_pinocchio_model():
    n_seg = 5
    p_markers = [jnp.array([[1.92, 0., 0.]]).T]

    dlo_params = models.load_aluminium_rod_params()
    rfem_params = models.RFEMParameters(
        n_seg, dlo_params, jutils.JointType.FREE, p_markers
    )
    model, _ = models.create_rfem_pinocchio_model(rfem_params)
    assert model.nq == 2*n_seg + 6
    assert model.frames == 1 + 1 # one for the world
    assert model.njoints == n_seg + 1


if __name__ == '__main__':
    test_create_custom_model()
    test_create_rfem_pinocchio_model()