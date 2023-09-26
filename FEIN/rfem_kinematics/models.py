import yaml
import numpy as np
import pinocchio as pin
import hppfcl as fcl
from collections import namedtuple
from typing import List
import jax.numpy as jnp

import FEIN.rfem_kinematics.rfem as rfem
from FEIN.rfem_kinematics.visualization import visualize_robot
import FEIN.utils.kinematics as jutils


class RFEMParameters:
    def __init__(self, 
        n_seg: int, 
        dlo_params: rfem.DLOParameters,
        base_joint_type: jutils.JointType = jutils.JointType.FREE,
        p_markers: List = None
    ) -> None:
        self.n_seg = n_seg
        self.base_joint_type = base_joint_type

        if p_markers is not None:
            self.marker_positions = p_markers
        else:
            self.marker_positions = []

        L = dlo_params.length
        self.lengths = rfem.compute_rfe_lengths(L, n_seg)
        self.m, self.rc, self.I = rfem.compute_rfe_inertial_parameters(n_seg, dlo_params)
        self.joint_force_callbacks = [lambda x, dx : 0.*x for _ in range(n_seg+1)]

        self.k_, self.d_ = rfem.compute_rfe_sde_parameters(n_seg, dlo_params)

    def set_marker_positions(self, m_positions):
        self.marker_positions = m_positions

    def set_masses(self, masses):
        self.m = masses

    def set_coms(self, rc):
        self.rc = rc

    def set_inertias(self, I):
        self.I = I

    def set_joint_force_callbacks(self, jfc):
        self.joint_force_callbacks = jfc

    def compute_jforce_callbacks_from_dlo_params(self):
        # Zero because the first joint is not part of the rfem and has zero stiffness
        self.joint_force_callbacks = (
            [lambda x, dx: 0. * x] + 
            [lambda x, dx: ki * x + di * dx for ki, di in zip(self.k_, self.d_)]
        )


def create_setup_pinocchio_model(rfem_params: RFEMParameters, add_ee_ref_joint: bool = False):
    base_joint_constructor = {
        jutils.JointType.U_ZY: create_universal_joint(),
        jutils.JointType.P_XYZ: pin.JointModelTranslation(),
        jutils.JointType.FREE: create_free_joint()
    }

    # Load panda model
    urdf_path = 'panda_description/panda_arm.urdf'
    model, cmodel, vmodel = pin.buildModelsFromUrdf(urdf_path)
    parent_frame_id = model.getFrameId('table_link')
    parent_joint_id = 0

    # Add rfem to the model
    rod_radius_viz = 0.035
    n_seg = rfem_params.n_seg
    rfe_lengths = np.asarray(rfem_params.lengths)
    rfe_m = [float(x) for x in rfem_params.m]
    rfe_rc = [np.asarray(x) for x in rfem_params.rc]
    rfe_I = [np.asarray(x) for x in rfem_params.I]
    p_markers = rfem_params.marker_positions
    base_joint = base_joint_constructor[rfem_params.base_joint_type]

    # Joint and body placement
    joint_placement = model.frames[parent_frame_id].placement # pin.SE3.Identity()
    body_placement = pin.SE3.Identity()
    body_placement.rotation = pin.rpy.rpyToMatrix(0., np.pi/2, 0.)
    
    # Implement universal joint by combining two revolute joints
    universal_joint = create_universal_joint()
    jtypes = [base_joint] + [universal_joint]*n_seg
    jids = []
    for k, (jtype, lxk, mk, rck, Ik) in enumerate(zip(jtypes, rfe_lengths, rfe_m, rfe_rc, rfe_I)):
        # Add joint to the model
        joint_name = 'rfem_joint_' + str(k+1)
        joint_id = model.addJoint(
            parent_joint_id,
            jtype,
            joint_placement,
            joint_name
        )
        jids.append(joint_id)
        
        body_inertia = pin.Inertia(mk, rck, Ik)
        body_placement.translation[0] = lxk
        model.appendBodyToJoint(joint_id, body_inertia, pin.SE3.Identity())

        # Define geometry for visualizzation
        geom_name = "rfe_" + str(k+1)
        shape = fcl.Cylinder(rod_radius_viz, body_placement.translation[0])
        shape_placement = body_placement.copy()
        shape_placement.translation[0] /= 2.
        geom_obj = pin.GeometryObject(
            geom_name, joint_id, shape, shape_placement
        )
        geom_obj.meshColor = np.array([0.85, 0.85, 0.85, 1.]) 
        vmodel.addGeometryObject(geom_obj)
        cmodel.addGeometryObject(geom_obj)

        # Adding the next joint
        # NOTE transformation from parent to joint assumed I
        parent_joint_id = joint_id
        joint_placement = pin.SE3.Identity()
        joint_placement.translation[0] = lxk

    fparents = rfem.compute_marker_frames_parent_joints(rfe_lengths, p_markers)
    fplacements = rfem.compute_marker_frames_placements(rfe_lengths, fparents, p_markers)
    for k, (pj_idx, f_pos) in enumerate(zip(fparents, fplacements)):
        # Attach frame in the middle of the link
        frame_name = 'marker_' + str(k+1)
        frame_placement = pin.SE3.Identity()
        frame_placement.translation = np.array(f_pos)
        frame = pin.Frame(
            frame_name, jids[pj_idx], jids[pj_idx], frame_placement, pin.FrameType.OP_FRAME
        )
        model.addFrame(frame)

    if add_ee_ref_joint:
        joint_parent_id = 0
        joint_placement = model.frames[parent_frame_id].placement
        joint_name = 'ee_ref_joint'
        joint_id = model.addJoint(
            joint_parent_id,
            pin.JointModelTranslation(),
            joint_placement,
            joint_name
        )

        body_inertia = pin.Inertia.Zero()
        model.appendBodyToJoint(joint_id, body_inertia, pin.SE3.Identity())

        geom_name = "ee_ref"
        shape = fcl.Sphere(0.015)
        shape_placement = pin.SE3.Identity()
        geom_obj = pin.GeometryObject(geom_name, joint_id, shape, shape_placement)
        geom_obj.meshColor = np.array([0.839, 0.075, 0.075, 1.])
        vmodel.addGeometryObject(geom_obj)
        cmodel.addGeometryObject(geom_obj)

    return model, cmodel, vmodel


def create_rfem_pinocchio_model(rfem_params: RFEMParameters, add_ee_ref_joint: bool = False):
    base_joint_constructor = {
        jutils.JointType.U_ZY: create_universal_joint(),
        jutils.JointType.P_XYZ: pin.JointModelTranslation(),
        jutils.JointType.FREE: create_free_joint()
    }
    n_seg = rfem_params.n_seg
    rfe_lengths = np.asarray(rfem_params.lengths)
    rfe_m = [float(x) for x in rfem_params.m]
    rfe_rc = [np.asarray(x) for x in rfem_params.rc]
    rfe_I = [np.asarray(x) for x in rfem_params.I]
    p_markers = rfem_params.marker_positions
    base_joint = base_joint_constructor[rfem_params.base_joint_type]

    # Instantiate a model
    model = pin.Model()
    model.name = "MRFEM"
    geom_model = pin.GeometryModel()
    parent_id = 0

    # Create base
    r_base = 0.015
    rod_radius_viz = 0.035# 0.006
    base_radius = r_base
    shape_base = fcl.Sphere(base_radius)
    base_placement = pin.SE3.Identity()
    geom_base = pin.GeometryObject("base", 0, shape_base, base_placement)
    geom_base.meshColor = np.array([1.,0.1,0.1,1.])
    geom_model.addGeometryObject(geom_base)

    # Joint and body placement
    joint_placement = pin.SE3.Identity()
    body_placement = pin.SE3.Identity()
    body_placement.rotation = pin.rpy.rpyToMatrix(0., np.pi/2, 0.)
    
    # Implement universal joint by combining two revolute joints
    universal_joint = create_universal_joint()
    jtypes = [base_joint] + [universal_joint]*n_seg
    jids = []
    for k, (jtype, lxk, mk, rck, Ik) in enumerate(zip(jtypes, rfe_lengths, rfe_m, rfe_rc, rfe_I)):
        # Add joint to the model
        joint_name = 'universal_joint_' + str(k+1)
        joint_id = model.addJoint(
            parent_id,
            jtype,
            joint_placement,
            joint_name
        )
        jids.append(joint_id)

        body_inertia = pin.Inertia(mk, rck, Ik)
        body_placement.translation[0] = lxk
        model.appendBodyToJoint(joint_id, body_inertia, pin.SE3.Identity())

        # Define geometry for visualizzation
        geom_name = "rfe_" + str(k+1)
        shape = fcl.Cylinder(rod_radius_viz, body_placement.translation[0])
        shape_placement = body_placement.copy()
        shape_placement.translation[0] /= 2.
        geom_obj = pin.GeometryObject(geom_name, joint_id, shape, shape_placement)
        geom_obj.meshColor = np.array([0.,0.,0.,1.])
        geom_model.addGeometryObject(geom_obj)

        # Adding the next joint
        # NOTE transformation from parent to joint assumed I
        parent_id = joint_id
        joint_placement = pin.SE3.Identity()
        joint_placement.translation[0] = lxk

    fparents = rfem.compute_marker_frames_parent_joints(rfe_lengths, p_markers)
    fplacements = rfem.compute_marker_frames_placements(rfe_lengths, fparents, p_markers)
    for k, (pj_idx, f_pos) in enumerate(zip(fparents, fplacements)):
        # Attach frame in the middle of the link
        frame_name = 'marker_' + str(k+1)
        frame_placement = pin.SE3.Identity()
        frame_placement.translation = np.array(f_pos)
        frame = pin.Frame(frame_name, jids[pj_idx], jids[pj_idx], frame_placement, pin.FrameType.OP_FRAME)
        model.addFrame(frame)

    if add_ee_ref_joint:
        joint_name = 'ee_ref_joint'
        joint_id = model.addJoint(
            0,
            pin.JointModelTranslation(),
            pin.SE3.Identity(),
            joint_name
        )

        body_inertia = pin.Inertia.Zero()
        model.appendBodyToJoint(joint_id, body_inertia, pin.SE3.Identity())

        geom_name = "ee_ref"
        shape = fcl.Sphere(0.01)
        shape_placement = pin.SE3.Identity()
        geom_obj = pin.GeometryObject(geom_name, joint_id, shape, shape_placement)
        geom_obj.meshColor = np.array([1.,0.1,0.1,1.])
        geom_model.addGeometryObject(geom_obj)

    return model, geom_model


def create_universal_joint(axis_1: str = 'Z', axis_2: str = 'Y'):
    """ Creates universal joint using Composite joint
    """
    name_to_impl = {
        'Z': pin.JointModelRZ(),
        'Y': pin.JointModelRY(),
        'X': pin.JointModelRX(),
    }

    universal_joint = pin.JointModelComposite()
    universal_joint.addJoint(name_to_impl[axis_1])
    universal_joint.addJoint(name_to_impl[axis_2])
    return universal_joint


def create_free_joint():
    """ Creates free joint using Composite joint
    """
    free_joint = pin.JointModelComposite()
    free_joint.addJoint(pin.JointModelTranslation())
    free_joint.addJoint(pin.JointModelSphericalZYX())
    return free_joint



""" For robot description named tuple is used, because it is more neat
in terms of accessing fields and it's immutable!
n_bodies -- number of bodies
n_joints -- number of joints
n_q -- the dimension of the configuration vector
n_frames -- number of frames
jnqs[i] -- number of elements in the joint config vector
jtypes[i] -- joint type of joint i
jparents[i] -- parent link of joint i
jplacements[jparents[i]] -- transfoormation from i joints parent to joint axes
fparents[i] -- parent joint of the frame i
fplacements[fparents[i]] -- transformation from frames parent to frame
inertias[i] -- inertia pcreate_nlink_spherical_pendulumarameters of link i
jforcecallbacks[i] -- force callback function of joint i

NOTE compared to classical robot data structure given by Featherstone
here parents[i] can rather be refered to frames and it contains as a
last element the transformation from the last joint to the end-effector!
"""
RobotDescription = namedtuple(
    'Robot',
    ['n_bodies', 'n_joints', 'n_q', 'n_frames',
     'jtypes', 'jnqs', 'jparents', 'jplacements',
     'fparents', 'fplacements',
     'inertias', 'jforcecallbacks']
)


def create_rfem_custom_model(rfem_params: RFEMParameters):
    n_seg = rfem_params.n_seg
    base_joint_type = rfem_params.base_joint_type
    rfe_m = rfem_params.m
    rfe_rc = rfem_params.rc
    rfe_I = rfem_params.I
    jforcecallbacks = rfem_params.joint_force_callbacks
    p_markers = rfem_params.marker_positions

    jparents = [-1] + [k for k in range(n_seg)]
    jtypes = [base_joint_type] + [jutils.JointType.U_ZY for _ in range(n_seg)]
    jnqs = [jutils.JOINTTYPE_TO_NQ[x] for x in jtypes]

    jlxs = rfem.compute_sde_joint_placements(rfem_params.lengths, frame = 'parent')
    jplacements = ([{'T': jutils.Rp2Trans(jnp.eye(3), jlxs[0])}] + # from base to first joint
        [{'T': jutils.Rp2Trans(jnp.eye(3), jp_k)} for jp_k in jlxs[1:]]
    )
    inertias = [{'I': jutils.inertia_at_joint(R_ab=jnp.eye(3), p_ba=rck, m=mk, I_b=Ik)}
                for mk, rck, Ik in zip(rfe_m, rfe_rc, rfe_I)]

    # Describe frames which correspond to sensor placements
    fparents = rfem.compute_marker_frames_parent_joints(rfem_params.lengths, p_markers)
    flxs = rfem.compute_marker_frames_placements(rfem_params.lengths, fparents, p_markers)
    fplacements = [{'T': jutils.Rp2Trans(jnp.eye(3), fp_k)} for fp_k in flxs]

    # Descriptive params
    n_bodies = n_seg+1
    n_joints = n_seg+1
    n_q = sum(jnqs)
    n_frames = len(fparents)

    model = RobotDescription(
        n_bodies,
        n_joints,
        n_q,
        n_frames,
        jtypes,
        jnqs,
        jparents,
        jplacements,
        fparents,
        fplacements,
        inertias,
        jforcecallbacks
    )
    return model


def load_aluminium_rod_params():
    """ Loads parameters for alimunium rod
    """
    path_to_yaml = 'FEIN/rfem_kinematics/configs/alrod-physical-params.yaml'
    return load_dlo_params_from_yaml(path_to_yaml)


def load_dlo_params_from_yaml(path: str):
    with open(path, 'r') as file:
        data = yaml.safe_load(file)
    return rfem.DLOParameters(**data)


def main():
    n_seg = 5
    P_markers = [1.92]

    dlo_params = load_aluminium_rod_params()
    rfem_params = RFEMParameters(n_seg, dlo_params, jutils.JointType.FREE, P_markers)
    model, geom_model = create_rfem_pinocchio_model(rfem_params)
    visual_model = geom_model

    custom_model = create_rfem_custom_model(rfem_params)

    q_b = np.array([[0., 0., 0.5, 0., 0., 0.]]).T
    q_sde = np.tile(np.array([[0., np.pi/(10*n_seg)]]).T, [n_seg,1])
    q = np.vstack((q_b, q_sde))
   
    visualize_robot(np.tile(q.T, [25,1]), 1, 2, model, geom_model)


if __name__ == "__main__":
    main()