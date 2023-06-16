# FEIN: finite element inspired networks

## Problem statement
Our primary focus lies in modeling the highly nonlinear dynamics of deformable linear objects, including cables, ropes, and rods, based on limited observations. An illustrative example showcasing the typical motion patterns we seek to predict is presented below.

<p align="center">
  <img src=https://github.com/shamilmamedov/FEIN/assets/59015432/df538094-14b2-4cd0-b800-abbecb2f02f0 alt="Image" />
</p>


The setup we used in this work to generate the dataset consists of: a 1.92 m long and 0.04 m thick aluminum rod that is rigidly connected to a Franka Panda robot arm; two marker frames attached to the DLO's start and end; a "Vicon" camera-based IPS records the motion of these frames (see the figure below). Inputs to the latent dynamic model are positions, velocities, and accelerations of the DLO's start (Panda's end-effector), while the partial observations or outputs are the DLO's end position and velocity. In essence, the model of the system should consist of:

- An encoder that maps a pair of observations and external inputs into the latent state.
- A latent dynamics model that maps the current state to the next one.
- A decoder that maps latent states and external inputs into observations.

![front page image v9-1](https://github.com/shamilmamedov/FEIN/assets/59015432/6350db74-2718-48da-9112-9b4047829203)

## Proposed method
Although purely machine learning (ML) methods can address the problem, they have a number of limitations. First, the latent state is often not interpretable and lacks physical meaning. Second, ML methods only predict the observations—the DLO's end position and velocity—without providing any information about the shape of the DLO. Rough shape reconstruction of the DLO is important for DLO manipulation in an environment with obstacles.

We propose approximating the DLO as a serial chain of rigid bodies connected via passive elastic joints, similar to the approach used in rigid FEM (RFEM). Utilizing the forward kinematics of this serial chain as a decoder makes the latent states to become physically meaningful and represent the generalized coordinates of the serial chain. Furthermore, this decoder enables the reconstruction of the shape of the DLO.

To further enhance the physical consistency of the DLO's shape, we apply regularization to the latent states. From a physical standpoint, this proposed regularization can be interpreted as minimizing kinetic and potential energy.

![fein v2-1](https://github.com/shamilmamedov/FEIN/assets/59015432/8d06c274-0fe0-4dce-9ce6-3958286c7d40)

## Running code
To use the code, you first need to clone the repository and install FEIN by running `pip install -e .`

To train a model without logging the results into [Weights and Biases](https://wandb.ai/site), execute `python training/run_experiment.py --config {config_file} --wandb_mode disabled`. You can choose one of the config files from the `training/experiment_configs` folder or create your own by following the same template. If you wish to try different encoder, decoder, or latent dynamics models, you will need to implement them and integrate them into the existing training routine in the `training/run_experiment.py` script. Please note that training a Neural ODE may take a significant amount of time.
