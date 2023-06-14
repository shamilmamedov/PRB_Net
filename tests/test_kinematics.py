import time
import numpy as np
import pinocchio as pin
import jax.numpy as jnp
import pytest
from pinocchio import JointModelTranslation

from jax.config import config
config.update("jax_enable_x64", True)


import FEIN.rfem_kinematics.algorithms as algos
from FEIN.rfem_kinematics import models
from FEIN.utils.kinematics import Trans2Rp, JointType



N_TESTS = 50
N_SEGS = [2, 4, 6]
P_markers = [0.3, 1.0, 2.42]
BASE_JOINT_TYPES = [JointType.U_ZY, JointType.P_XYZ, JointType.FREE]

def test_fwd_kinematics_rfem():
    """ Tests forward kinematics algorithm that is under the
    hood of marker position calculation algorithm
    """
    def inner_test_loop(m_pin, m_own):
        d_pin = m_pin.createData()
        for _ in range(N_TESTS):
            q_rnd = np.random.uniform(-np.pi, np.pi, size=(m_pin.nq,1))
            pin.forwardKinematics(m_pin, d_pin, q_rnd)
            pin.updateFramePlacements(m_pin, d_pin)
            
            _, o_T_f = algos.fwd_kinematics(m_own, q_rnd)
            p_markers = algos.compute_markers_positions(m_own, q_rnd)
            p_markers2 = algos.compute_markers_positions_and_velocities(
                m_own, q_rnd, jnp.zeros_like(q_rnd)
            )[0]
            for k in range(3):
                R_pin, p_pin = d_pin.oMf[k+1].rotation, d_pin.oMf[k+1].translation
                
                o_R_f, o_p_f = Trans2Rp(o_T_f[k])

                np.testing.assert_array_almost_equal(R_pin, o_R_f, decimal=5)
                np.testing.assert_array_almost_equal(p_pin, o_p_f.flatten(), decimal=5)
                np.testing.assert_array_almost_equal(p_pin, p_markers[k], decimal=5)
                np.testing.assert_array_almost_equal(p_pin, p_markers2[k], decimal=5)
    
    dlo_params = models.load_aluminium_rod_params()

    for bjoint in BASE_JOINT_TYPES:
        for n_seg in N_SEGS:
            rfem_params = models.RFEMParameters(n_seg, dlo_params, bjoint, P_markers)

            # Pin model
            model, _ = models.create_rfem_pinocchio_model(rfem_params)

            # Get custom model description
            custom_model = models.create_rfem_custom_model(rfem_params)

            inner_test_loop(model, custom_model)


def test_fwd_velocity_kinematics_rfem():
    dlo_params = models.load_aluminium_rod_params()
    for bjoint in BASE_JOINT_TYPES:
        for n_seg in N_SEGS:
            rfem_params = models.RFEMParameters(n_seg, dlo_params, bjoint, P_markers)

            # Pin model
            pin_model, _ = models.create_rfem_pinocchio_model(rfem_params)
            pin_data = pin_model.createData()

            # Get custom model description
            custom_model = models.create_rfem_custom_model(rfem_params)

            for _ in range(N_TESTS):
                q_rnd = np.pi*np.random.uniform(-1, 1, size=(pin_model.nq,1))
                dq_rnd = 2*np.pi*np.random.uniform(-1, 1, size=(pin_model.nq, 1))

                pin.forwardKinematics(pin_model, pin_data, q_rnd, dq_rnd)
                pin.updateFramePlacements(pin_model, pin_data)

                Vj, o_Vf = algos.fwd_velocity_kinematics(custom_model, q_rnd, dq_rnd)
                dp_markers = algos.compute_markers_velocities(custom_model, q_rnd, dq_rnd)
                dp_markers2 = algos.compute_markers_positions_and_velocities(custom_model, q_rnd, dq_rnd)[1]

                for k in range(3):
                    v_ = pin.getFrameVelocity(pin_model, pin_data, 
                                pin_model.getFrameId(f'marker_{k+1}'), 
                                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                    v_pin = np.concatenate((v_.angular, v_.linear))[:, None]

                    np.testing.assert_array_almost_equal(o_Vf[k], v_pin)
                    np.testing.assert_array_almost_equal(dp_markers[k], v_.linear)
                    np.testing.assert_array_almost_equal(dp_markers2[k], v_.linear)


if __name__ == "__main__":
    test_fwd_kinematics_rfem()
    test_fwd_velocity_kinematics_rfem()
    

    

    
