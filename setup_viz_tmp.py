import numpy as np
import pandas as pd

from FEIN.rfem_kinematics.models import (
    create_setup_pinocchio_model, load_aluminium_rod_params, RFEMParameters)
from FEIN.rfem_kinematics.visualization import visualize_robot
import FEIN.utils.kinematics as jutils


def main():
    n_seg = 7
    P_markers = [1.92]

    dlo_params = load_aluminium_rod_params()
    rfem_params = RFEMParameters(n_seg, dlo_params, jutils.JointType.FREE, P_markers)
    model, cmodel, vmodel = create_setup_pinocchio_model(rfem_params)

    dir = '/home/shamil/Desktop/phd/dataset/DLO-processed/pool-noodle/'
    traj = 'traj2.csv'
    traj_path = dir + traj

    axes = ['x', 'y', 'z']
    traj = pd.read_csv(traj_path)
    qp_col_names = [f'q_{i}' for i in range(1, 8)]
    pb_col_names = [f'p_b_{x}' for x in axes]
    phib_col_names = [f'phi_b_{x}' for x in axes]
    qb_col_names = pb_col_names + phib_col_names
    qp = traj[qp_col_names].values
    qb = traj[qb_col_names].values
    q = np.zeros((qp.shape[0], model.nq))
    q[:,:7] = qp
    q[:,7:13] = qb

    dt = 0.004
    n_replays = 1
    visualize_robot(q, dt, n_replays, model, cmodel, vmodel)#, 'meshcat')


if __name__ == '__main__':
    main()