
from FEIN.utils.data import load_trajs


def main():
    n_traj = [5]
    traj = load_trajs(n_traj, 'pool-noodle')[0]

    traj.plot('p_e')


if __name__ == '__main__':
    main()