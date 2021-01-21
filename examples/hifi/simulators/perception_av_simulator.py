"""A toy simulator of a scenario of an AV approaching a crosswalk where some pedestrians are crossing."""
import collections
import numpy as np
import os
import time
import pathlib
import pdb
import pickle
import hickle as hkl
from tqdm import tqdm
import scipy.stats
from scipy.ndimage import gaussian_filter

from examples.hifi.EnvironmentPrediction.Predictions.ParticleFilter.NewParticleInitialization import *
from examples.hifi.EnvironmentPrediction.Predictions.ParticleFilter.ParticlePrediction import *
from examples.hifi.EnvironmentPrediction.Predictions.ParticleFilter.ParticleAssignment import *
from examples.hifi.EnvironmentPrediction.Predictions.ParticleFilter.OccupancyPredictionUpdate import *
from examples.hifi.EnvironmentPrediction.Predictions.ParticleFilter.PersistentParticleUpdate import *
from examples.hifi.EnvironmentPrediction.Predictions.ParticleFilter.StatisticMoments import *
from examples.hifi.EnvironmentPrediction.Predictions.ParticleFilter.Resample import *

from examples.hifi.EnvironmentPrediction.Predictions.ParticleFilter.Particle import *
from examples.hifi.EnvironmentPrediction.Predictions.ParticleFilter.Grid import *
from examples.hifi.EnvironmentPrediction.Predictions.ParticleFilter.PlotTools import colorwheel_plot

# Define the class


class PerceptionAVSimulator():
    """A toy simulator of a scenario of an AV approaching a crosswalk where some pedestrians are crossing.

    The vehicle runs a modified version of the Intelligent Driver Model [1]_. The vehicle treats the closest
    pedestrian in the road as a car to follow. If no pedestrians are in the road, it attempts to maintain the
    desired speed. Noisy observations of the pedestrian are smoothed through an alpha-beta filter [2]_.

    A collision results if any pedestrian's x-distance and y-distance to the ego vehicle are less than the respective
    `min_dist_x` and `min_dist_y`.

    The origin is centered in the middle of the east/west lane and the north/south crosswalk.
    The positive x proceeds east down the lane, the positive y proceeds north across the crosswalk.

    Parameters
    ----------
    num_peds : int
        The number of pedestrians crossing the street.
    dt : float
        The length (in seconds) of each timestep.
    alpha : float
        The alpha parameter in the tracker's alpha-beta filter [2]_.
    beta : float
        The beta parameter in the tracker's alpha-beta filter [2]_.
    v_des : float
        The desired velocity, in meters per second,  for the ego vehicle to maintain
    delta : float
        The delta parameter in the IDM algorithm [1]_.
    t_headway : float
        The headway parameter in the IDM algorithm [1]_.
    a_max : float
        The maximum acceleration parameter in the IDM algorithm [1]_.
    s_min : float
        The minimum follow distance parameter in the IDM algorithm [1]_.
    d_cmf : float
        The maximum comfortable deceleration parameter in the IDM algorithm [1]_.
    d_max : float
        The maximum deceleration parameter in the IDM algorithm [1]_.
    min_dist_x : float
        The minimum x-distance between the ego vehicle and a pedestrian.
    min_dist_y : float
        The minimum y-distance between the ego vehicle and a pedestrian.
    car_init_x : float
        The initial x-position of the ego vehicle.
    car_init_y : float
        The initial y-position of the ego vehicle.

    References
    ----------
    .. [1] Treiber, Martin, Ansgar Hennecke, and Dirk Helbing.
        "Congested traffic states in empirical observations and microscopic simulations."
        Physical review E 62.2 (2000): 1805.
        `<https://journals.aps.org/pre/abstract/10.1103/PhysRevE.62.1805>`_
    .. [2] Rogers, Steven R. "Alpha-beta filter with correlated measurement noise."
        IEEE Transactions on Aerospace and Electronic Systems 4 (1987): 592-594.
        `<https://ieeexplore.ieee.org/abstract/document/4104388>`_
    """
    # Accept parameters for defining the behavior of the system under test[SUT]

    def __init__(self,
                 num_peds=1,
                 dt=0.1,
                 alpha=0.85,
                 beta=0.005,
                 v_des=11.17,
                 delta=4.0,
                 t_headway=1.5,
                 a_max=3.0,
                 s_min=4.0,
                 d_cmf=2.0,
                 d_max=9.0,
                 min_dist_x=2.5,
                 min_dist_y=1.4,
                 car_init_x=-35.0,
                 car_init_y=0.0,
                 use_tracker=True,
                 perception_type='basic',
                 perception_args=None,
                 noise_vector_size=10,
                 expert_trajectory=False,
                 ):

        # Constant hyper-params -- set by user
        self.c_num_peds = num_peds
        self.c_dt = dt
        self.c_alpha = alpha
        self.c_beta = beta
        self.c_v_des = v_des
        self.c_delta = delta
        self.c_t_headway = t_headway
        self.c_a_max = a_max
        self.c_s_min = s_min
        self.c_d_cmf = d_cmf
        self.c_d_max = d_max
        self.c_min_dist = np.array([min_dist_x, min_dist_y])
        self.c_car_init_x = car_init_x
        self.c_car_init_y = car_init_y
        # self.blackbox_sim_state = blackbox_sim_state

        # These are set by reset, not the user
        self._car = np.zeros((4))
        self._car_accel = np.zeros((2))
        self._peds = np.zeros((self.c_num_peds, 4))
        self._measurements = np.zeros((self.c_num_peds, 4))
        self._car_obs = np.zeros((self.c_num_peds, 4))
        self._env_obs = np.zeros((self.c_num_peds, 4))
        self._done = False
        self._reward = 0.0
        self._info = []
        self._step = 0
        self._path_length = 0
        # self._action = None
        self._action = np.array([0] * (3 * self.c_num_peds))
        self._first_step = True
        self.directions = np.random.randint(2, size=self.c_num_peds) * 2 - 1
        self.y = np.random.rand(self.c_num_peds) * 14 - 5
        self.x = np.random.rand(self.c_num_peds) * 4 - 2
        self._state = None
        self.noise_vector_size = noise_vector_size

        self.use_tracker = use_tracker
        self.perception_type = perception_type
        self.expert_trajectory = expert_trajectory

        if perception_args is None:
            perception_args = {}

        self.perception_args = perception_args

        if self.perception_type == 'sonar':
            if 'tolerance' not in self.perception_args:
                self.perception_args['tolerance'] = 0.5

            if 'ray_halfwidth' not in self.perception_args:
                self.perception_args['ray_halfwidth'] = 15

            if 'tolerance' not in self.perception_args:
                self.perception_args['tolerance'] = 0.5

            if 'tolerance' not in self.perception_args:
                self.perception_args['tolerance'] = 0.5

        if self.perception_type == 'dogma':
            if 'do_plot' not in self.perception_args:
                self.perception_args['do_plot'] = False

            if 'shape' not in self.perception_args:
                self.perception_args['shape'] = (128, 128)

            if 'p_B' not in self.perception_args:
                self.perception_args['p_B'] = 0.02

            if 'Vb' not in self.perception_args:
                self.perception_args['Vb'] = 2 * 10 ** 3

            if 'V' not in self.perception_args:
                self.perception_args['V'] = 2 * 10 ** 4

            if 'state_size' not in self.perception_args:
                self.perception_args['state_size'] = 4

            if 'alpha' not in self.perception_args:
                self.perception_args['alpha'] = 0.9

            if 'p_A' not in self.perception_args:
                self.perception_args['p_A'] = 1.0

            if 'T' not in self.perception_args:
                self.perception_args['T'] = 0.1

            if 'p_S' not in self.perception_args:
                self.perception_args['p_S'] = 0.99

            if 'scale_vel' not in self.perception_args:
                self.perception_args['scale_vel'] = 12.0

            if 'scale_acc' not in self.perception_args:
                self.perception_args['scale_acc'] = 2.0

            if 'process_pos' not in self.perception_args:
                self.perception_args['process_pos'] = 0.06

            if 'process_vel' not in self.perception_args:
                self.perception_args['process_vel'] = 2.4

            if 'process_acc' not in self.perception_args:
                self.perception_args['process_acc'] = 0.2

            if 'verbose' not in self.perception_args:
                self.perception_args['verbose'] = False

            if 'mS' not in self.perception_args:
                self.perception_args['mS'] = 1.0

            if 'epsilon' not in self.perception_args:
                self.perception_args['epsilon'] = 10.0

            if 'epsilon_occ' not in self.perception_args:
                self.perception_args['epsilon_occ'] = 0.5

            if 'num_beams' not in self.perception_args:
                self.perception_args['num_beams'] = 30

            if 'total_beam_width' not in self.perception_args:
                self.perception_args['total_beam_width'] = np.pi

            if 'beam_width' not in self.perception_args:
                self.perception_args['beam_width'] = self.perception_args['total_beam_width'] / self.perception_args['num_beams']

            if 'target_thickness' not in self.perception_args:
                self.perception_args['target_thickness'] = 1.0

            if 'grid_size' not in self.perception_args:
                self.perception_args['grid_size'] = 0.5

            if 'xshift' not in self.perception_args:
                self.perception_args['xshift'] = -35

            if 'yshift' not in self.perception_args:
                self.perception_args['yshift'] = -6.5

            if 'dogma_state_filepath' not in self.perception_args:
                self.perception_args['dogma_state_filepath'] = './dogma'

            self.dogma_state_filepath = pathlib.Path(self.perception_args['dogma_state_filepath'])
            self.dogma_state_filepath.mkdir(parents=True, exist_ok=True)

            self.coord_grid = create_map(self.perception_args['shape'][0] - 2,
                                         self.perception_args['shape'][1] - 2,
                                         grid_size=self.perception_args['grid_size'],
                                         xshift=self.perception_args['xshift'],
                                         yshift=self.perception_args['yshift'],
                                         beta=self.perception_args['beam_width'])

            self.dogma = DOGMA(do_plot=self.perception_args['do_plot'],
                         shape=self.perception_args['shape'],
                         p_B=self.perception_args['p_B'],
                         Vb=self.perception_args['Vb'],
                         V=self.perception_args['V'],
                         state_size=self.perception_args['state_size'],
                         alpha=self.perception_args['alpha'],
                         p_A=self.perception_args['p_A'],
                         T=self.perception_args['T'],
                         p_S=self.perception_args['p_S'],
                         scale_vel=self.perception_args['scale_vel'],
                         scale_acc=self.perception_args['scale_acc'],
                         process_pos=self.perception_args['process_pos'],
                         process_vel=self.perception_args['process_vel'],
                         process_acc=self.perception_args['process_acc'],
                         verbose=self.perception_args['verbose'],
                         mS=self.perception_args['mS'],
                         epsilon=self.perception_args['epsilon'],
                         epsilon_occ=self.perception_args['epsilon_occ'])


    def run_simulation(self, actions, s_0, simulation_horizon):
        """Run a full simulation given the AST solver's actions and initial conditions.

        Parameters
        ----------
        actions : list[array_like]
            A sequential list of actions taken by the AST Solver which deterministically control the simulation.
        s_0 : array_like
            An array specifying the initial conditions to set the simulator to.
        simulation_horizon : int
            The maximum number of steps a simulation rollout is allowed to run.

        Returns
        -------
        terminal_index : int
            The index of the action that resulted in a state in the goal set E. If no state is found
            terminal_index should be returned as -1.
        array_like
            An array of relevant simulator info, which can then be used for analysis or diagnostics.

        """
        # initialize the simulation
        path_length = 0
        self.reset(s_0)
        self._info = []

        simulation_horizon = np.minimum(simulation_horizon, len(actions))

        # Take simulation steps unbtil horizon is reached
        while path_length < simulation_horizon:
            # get the action from the list
            self._action = actions[path_length]

            # Step the simulation forward in time
            self.step_simulation(self._action)

            # check if a crash has occurred. If so return the timestep, otherwise continue
            if self.collision_detected():
                return path_length, np.array(self._info)
            path_length = path_length + 1

        # horizon reached without crash, return -1
        self._is_terminal = True
        return -1, np.array(self._info)

    def step_simulation(self, action):
        """
        Handle anything that needs to take place at each step, such as a simulation update or write to file.

        Parameters
        ----------
        action : array_like
            A 1-D array of actions taken by the AST Solver which deterministically control
            a single step forward in the simulation.

        Returns
        -------
        array_like
            An observation from the timestep, determined by the settings and the `observation_return` helper function.

        """
        # return None

        # get the action from the list
        self._action = action

        # move the peds
        self.update_peds()

        # move the car
        self._car = self.move_car(self._car, self._car_accel)

        # take new measurements and noise them
        if self.perception_type == 'dogma':
            noise = self._action.reshape((self.c_num_peds, self.noise_vector_size))[:, 2:self.noise_vector_size]
        else:
            noise = self._action.reshape((self.c_num_peds, self.noise_vector_size))[:, 2:self.noise_vector_size]
        self._measurements = self.sensors(self._peds, noise)

        # filter out the noise with an alpha-beta tracker
        self._car_obs = self.tracker(self._car_obs, self._measurements)

        # select the SUT action for the next timestep
        self._car_accel[0] = self.update_car(self._car_obs, self._car[0])

        # grab simulation state, if interactive
        self.observe()
        self.observation = np.ndarray.flatten(self._env_obs)
        # record step variables
        self.log()

        self._step += 1

        return self.observation

    def reset(self, s_0):
        """Resets the state of the environment, returning an initial observation.

        Parameters
        ----------
        s_0 : array_like
            The initial conditions to reset the simulator to.

        Returns
        -------
        array_like
            An observation from the timestep, determined by the settings and the `observation_return` helper function.
        """

        # initialize variables
        self._info = []
        self._step = 0
        self._path_length = 0
        self._is_terminal = False
        self.initial_conditions = s_0
        self._action = np.array([0] * (self.noise_vector_size * self.c_num_peds))
        self._first_step = True

        # Get v_des if it is sampled from a range
        v_des = self.initial_conditions[3 * self.c_num_peds]

        # initialize SUT location
        car_init_x = self.initial_conditions[3 * self.c_num_peds + 1]
        self._car = np.array([v_des, 0.0, car_init_x, self.c_car_init_y])

        # zero out the first SUT acceleration
        self._car_accel = np.zeros((2))

        # initialize pedestrian locations and velocities
        pos = self.initial_conditions[0:2 * self.c_num_peds]
        self.x = pos[0:self.c_num_peds * 2:2]
        self.y = pos[1:self.c_num_peds * 2:2]
        v_start = self.initial_conditions[2 * self.c_num_peds:3 * self.c_num_peds]
        self._peds[0:self.c_num_peds, 0] = np.zeros((self.c_num_peds))
        self._peds[0:self.c_num_peds, 1] = v_start
        self._peds[0:self.c_num_peds, 2] = self.x
        self._peds[0:self.c_num_peds, 3] = self.y

        # Calculate the relative position measurements
        self._measurements = self._peds
        self._env_obs = self._measurements
        self._car_obs = self._measurements

        # return the initial simulation state
        self.observation = np.ndarray.flatten(self._measurements)
        # self.observation = obs

        # if self.perception_type == 'dogma':



        return self.observation

    def collision_detected(self):
        """
        Returns whether the current state is in the goal set.

        Checks to see if any pedestrian's position violates both the `min_dist_x` and `min_dist_y` constraints.

        Returns
        -------
        bool
            True if current state is in goal set.
        """
        # calculate the relative distances between the pedestrians and the car
        dist = self._peds[:, 2:4] - self._car[2:4]

        # return True if any relative distance is within the SUT's hitbox and the car is still moving
        if (np.any(np.all(np.less_equal(abs(dist), self.c_min_dist), axis=1)) and
                self._car[0] > 0.5):
            return True

        return False

    def log(self):
        """
        Perform any logging steps.

        """
        # Create a cache of step specific variables for post-simulation analysis
        cache = np.hstack([0.0,  # Dummy, will be filled in with trial # during post processing in save_trials.py
                           self._step,
                           np.ndarray.flatten(self._car),
                           np.ndarray.flatten(self._peds),
                           np.ndarray.flatten(self._action),
                           np.ndarray.flatten(self._car_obs),
                           0.0])
        self._info.append(cache)
        # self._step += 1

    def sensors(self, peds, noise):
        """Get a noisy observation of the pedestrians' locations and velocities.

        Parameters
        ----------
        peds : array_like
            Positions and velocities of the pedestrians.
        noise : array_like
            Noise to add to the positions and velocities of the pedestrians.

        Returns
        -------
        array_like
            Noisy observation of the pedestrians' locations and velocities.

        """
        if self.perception_type == 'sonar':
            measurements = peds[:, 2:self.noise_vector_size] + noise
        elif self.perception_type == 'dogma':
            relative_dist = self._peds - self._car
            relative_norm = np.linalg.norm(relative_dist[:, 2:])
            relative_angles = np.arctan2(relative_dist[:, 3], relative_dist[:, 2])

            # pdb.set_trace()
            # beam_angles = car_theta + (np.arange(num_beams)*np.pi/num_beams - np.pi/2)
            car_theta = np.arctan2(self._car[1], self._car[0])

            beam_angles = car_theta + (
                    np.arange(self.perception_args['num_beams']) *
                    (self.perception_args['total_beam_width']) /
                    (self.perception_args['num_beams'] - 1)
                    - self.perception_args['total_beam_width'] / 2)
            # Wrap to +pi / - pi
            beam_angles[beam_angles > np.pi] -= 2. * np.pi
            beam_angles[beam_angles < -np.pi] += 2. * np.pi

            measurements = np.concatenate([np.ones((1, self.perception_args['num_beams'])) * 100,
                                   beam_angles.reshape(1, -1)])

            measurements[0][np.where(np.logical_and(relative_angles > beam_angles - self.perception_args['beam_width'],
                                            relative_angles < beam_angles + self.perception_args['beam_width']))] = relative_norm + noise

            # measurements[np.where(measurements[])]
        else:
            measurements = peds + noise

        return measurements

    def tracker(self, estimate_old, measurements):
        """An alpha-beta filter to smooth noisy observations into an estimate of pedestrian state.

        Parameters
        ----------
        estimate_old : array_like
            The smoothed state estimate from the previous timestep.
        measurements : array_like
            The noisy observation of pedestrian state from the current timestep.

        Returns
        -------
        array_like
            The smoothed state estimate of pedestrian state from the current timestep.
        """
        if not self.use_tracker:
            return measurements

        if self.perception_type == 'sonar':
            # Occupancy grid update
            # Calculate which regions should be updated as "occupied"
            relative_dist = self._peds - self._car
            relative_norm = np.linalg.norm(relative_dist)
            relative_angles = np.arctan2(relative_dist[:,1],  relative_dist[:,0])

            occupied_regions = relative_norm[np.where(relative_angles <= self.perception_args['ray_halfwidth'])]

            # Update the grid
            self.log_prob_map = np.zeros((xsize, ysize))
            self.grid_position_m = np.array(
                    [np.tile(np.arange(0, xsize * grid_size, grid_size)[:, None], (1, ysize)),
                    np.tile(np.arange(0, ysize * grid_size, grid_size)[:, None].T, (xsize, 1))])

        elif self.perception_type == 'dogma':
            car_x = self._car[2]
            car_y = self._car[3]
            car_theta = np.arctan2(self._car[1], self._car[0])
            state = np.array([car_x, car_y, car_theta])
            grid = self.dogma.lidar_to_grid(pose=state, lidar=measurements.T, grid_position_m=self.coord_grid, beta=self.perception_args['beam_width'],
                                       alpha=self.perception_args['target_thickness'])  # update the map

            # print(grid.shape)
            self.dogma.update(grid[np.newaxis, ...])

            m_occ, m_free = self.dogma.DOGMA[-1][0:2, :, :]
            vel_x, vel_y = self.dogma.DOGMA[-1][2:4, :, :] * self.perception_args['grid_size']
            meas_grid = self.dogma.DOGMA[-1][4, :, :]
            min_occ_prob = 0.50
            peds = np.array([vel_x[np.where((m_occ >= min_occ_prob) & (grid != 3))],
                             vel_y[np.where((m_occ >= min_occ_prob) & (grid != 3))],
                             self.coord_grid[0, :, :][np.where((m_occ >= min_occ_prob) & (grid != 3))],
                             self.coord_grid[1, :, :][np.where((m_occ >= min_occ_prob) & (grid != 3))],
                             ]).T

            observation = np.mean(peds, axis=0)
            observation = observation[np.newaxis, :]

        else:
            # Alpha-Beta Filter
            observation = np.zeros_like(estimate_old)

            observation[:, 0:2] = estimate_old[:, 0:2]
            observation[:, 2:4] = estimate_old[:, 2:4] + self.c_dt * estimate_old[:, 0:2]
            residuals = measurements[:, 2:4] - observation[:, 2:4]

            observation[:, 2:4] += self.c_alpha * residuals
            observation[:, 0:2] += self.c_beta / self.c_dt * residuals

        return observation

    def update_car(self, obs, v_car):
        """Calculate the ego vehicle's acceleration.

        Parameters
        ----------
        obs : array_like
            Smoothed estimate of pedestrian state from the `tracker`.
        v_car : float
            Current velocity of the ego vehicle.

        Returns
        -------
        float
            The acceleration of the ego vehicle.

        """
        cond = np.repeat(np.resize(np.logical_and(obs[:, 3] > -1.5, obs[:, 3] < 4.5), (self.c_num_peds, 1)), 4, axis=1)
        in_road = np.expand_dims(np.extract(cond, obs), axis=0)

        if in_road.size != 0:
            mins = np.argmin(in_road.reshape((-1, 4)), axis=0)
            v_oth = obs[mins[3], 0]
            s_headway = obs[mins[3], 2] - self._car[2]
            s_headway = max(10 ** -6, abs(s_headway)) * np.sign(s_headway)  # avoid div by zero error later

            del_v = v_oth - v_car
            s_des = self.c_s_min + v_car * self.c_t_headway - v_car * del_v / (2 * np.sqrt(self.c_a_max * self.c_d_cmf))
            if self.c_v_des > 0.0:
                v_ratio = v_car / self.c_v_des
            else:
                v_ratio = 1.0

            a = self.c_a_max * (1.0 - v_ratio ** self.c_delta - (s_des / s_headway) ** 2)

        else:
            del_v = self.c_v_des - v_car
            a = del_v

        if np.isnan(a):
            pdb.set_trace()
        # pdb.set_trace()
        return np.clip(a, -self.c_d_max, self.c_a_max)

    def move_car(self, car, accel):
        """Update the ego vehicle's state.

        Parameters
        ----------
        car : array_like
            The ego vehicle's state: [x-velocity, y-velocity, x-position, y-position].
        accel : float
            The ago vehicle's acceleration.

        Returns
        -------
        array_like
            An updated version of the ego vehicle's state.

        """
        car[2:4] += self.c_dt * car[0:2]
        car[0:2] += self.c_dt * accel
        return car

    def update_peds(self):
        """Update the pedestrian's state.

        """
        # Update ped state from actions
        action = self._action.reshape((self.c_num_peds, self.noise_vector_size))[:, 0:2]

        mod_a = np.hstack((action,
                           self._peds[:, 0:2] + 0.5 * self.c_dt * action))
        if np.any(np.isnan(mod_a)):
            pdb.set_trace()

        self._peds += self.c_dt * mod_a
        # Enforce max abs(velocity) on pedestrians
        self._peds[:, 0:2] = np.clip(self._peds[:, 0:2], a_min=[-4.5, -4.5], a_max=[4.5, 4.5])
        if np.any(np.isnan(self._peds)):
            pdb.set_trace()

    def observe(self):
        """Get the ground truth state of the pedestrian relative to the ego vehicle.

        """
        self._env_obs = self._peds - self._car

    def get_ground_truth(self):
        """Clones the ground truth simulator state.

        Returns
        -------
        dict
            A dictionary of simulator state variables.
        """
        # import pdb
        # pdb.set_trace()
        state_dict = {'step': self._step,
                'path_length': self._path_length,
                'is_terminal': self._is_terminal,
                'car': self._car,
                'car_accel': self._car_accel,
                'peds': self._peds,
                'car_obs': self._car_obs,
                'action': self._action,
                'initial_conditions': self.initial_conditions,}

        # if self.perception_type == 'dogma':
        #     var_x_vel_0 = np.zeros(self.dogma.shape)
        #     var_x_vel_1 = np.zeros(self.dogma.shape)
        #     var_y_vel_0 = np.zeros(self.dogma.shape)
        #     var_y_vel_1 = np.zeros(self.dogma.shape)
        #     covar_xy_vel_0 = np.zeros(self.dogma.shape)
        #     covar_xy_vel_1 = np.zeros(self.dogma.shape)
        #     DOGMA_0 = np.zeros((5, self.dogma.shape[0], self.dogma.shape[1]))
        #     DOGMA_1 = np.zeros((5, self.dogma.shape[0], self.dogma.shape[1]))
        #
        #     if len(self.dogma.var_x_vel) >= 1:
        #         var_x_vel_0 = self.dogma.var_x_vel[0]
        #     if len(self.dogma.var_x_vel) >= 2:
        #         var_x_vel_1 = self.dogma.var_x_vel[1]
        #
        #     if len(self.dogma.var_y_vel) >= 1:
        #         var_y_vel_0 = self.dogma.var_y_vel[0]
        #     if len(self.dogma.var_y_vel) >= 2:
        #         var_y_vel_1 = self.dogma.var_y_vel[1]
        #
        #     if len(self.dogma.covar_xy_vel) >= 1:
        #         covar_xy_vel_0 = self.dogma.covar_xy_vel[0]
        #     if len(self.dogma.covar_xy_vel) >= 2:
        #         covar_xy_vel_1 = self.dogma.covar_xy_vel[1]
        #
        #     if len(self.dogma.DOGMA) >= 1:
        #         DOGMA_0 = self.dogma.DOGMA[0]
        #     if len(self.dogma.DOGMA) >= 2:
        #         DOGMA_1 = self.dogma.DOGMA[1]
        #
        #     state_dict['var_x_vel_0'] = var_x_vel_0
        #     state_dict['var_x_vel_1'] = var_x_vel_1
        #     state_dict['var_y_vel_0'] = var_y_vel_0
        #     state_dict['var_y_vel_1'] = var_y_vel_1
        #     state_dict['covar_xy_vel_0'] = covar_xy_vel_0
        #     state_dict['covar_xy_vel_1'] = covar_xy_vel_1
        #     state_dict['DOGMA_0'] = DOGMA_0
        #     state_dict['DOGMA_1'] = DOGMA_1

        return state_dict

    def set_ground_truth(self, in_simulator_state):
        """Sets the simulator state variables.

        Parameters
        ----------
        in_simulator_state : dict
            A dictionary of simulator state variables.
        """
        in_simulator_state.copy()

        self._step = in_simulator_state['step']
        self._path_length = in_simulator_state['path_length']
        self._is_terminal = in_simulator_state['is_terminal']
        self._car = in_simulator_state['car']
        self._car_accel = in_simulator_state['car_accel']
        self._peds = in_simulator_state['peds']
        self._car_obs = in_simulator_state['car_obs']
        self._action = in_simulator_state['action']
        self.initial_conditions = np.array(in_simulator_state['initial_conditions'])

        self.observe()
        self.observation = self._env_obs

        # self.dogma.var_x_vel[0] = in_simulator_state['var_x_vel_0']
        # self.dogma.var_x_vel[1] = in_simulator_state['var_x_vel_1'].reshape(self.dogma.shape)
        # self.dogma.var_y_vel[0] = in_simulator_state['var_y_vel_0'].reshape(self.dogma.shape)
        # self.dogma.var_y_vel[1] = in_simulator_state['var_y_vel_1'].reshape(self.dogma.shape)
        # self.dogma.covar_xy_vel[0] = in_simulator_state['covar_xy_vel_0'].reshape(self.dogma.shape)
        # self.dogma.covar_xy_vel[1] = in_simulator_state['covar_xy_vel_1'].reshape(self.dogma.shape)
        # self.dogma.DOGMA[0] = in_simulator_state['DOGMA_0']
        # self.dogma.DOGMA[1] = in_simulator_state['DOGMA_1'].reshape((5, self.dogma.shape[0], self.dogma.shape[1]))

    def clone_state(self):
        """Clone the simulator state for later resetting.

        This function is used in conjunction with `restore_state` for Go-Explore and Backwards Algorithm
        to do their deterministic resets.

        Returns
        -------
        array_like
            An array of all the simulation state variables.

        """
        # self.state_vars = collections.OrderedDict([('step', 1),
        #                                            ('b', 2),
        #                                            ('c', 3)])
        #
        # # state_index = 0
        # state = np.array([])
        # for var_name in self.state_vars:
        #     val = getattr(self, var_name)
        #     if isinstance(val, np.ndarray):
        #         val = val.flatten()
        #     else:
        #         val = np.array([val])
        #
        #     if state.size == 0:
        #         state = val
        #     else:
        #         np.concatenate((state, val), axis=0)
        #
        # return state

        # Get the ground truth state from the toy simulator
        simulator_state = self.get_ground_truth()

        if self.perception_type == 'dogma' and self.expert_trajectory:
            # Save DOGMA state
            dogma_state_filename = 'step_' + str(int(simulator_state['step']))
            self.dogma.clone_state(filepath=self.dogma_state_filepath, filename=dogma_state_filename)

        return np.concatenate((np.array([simulator_state['step']]),
                               np.array([simulator_state['path_length']]),
                               np.array([int(simulator_state['is_terminal'])]),
                               simulator_state['car'],
                               simulator_state['car_accel'],
                               simulator_state['peds'].flatten(),
                               simulator_state['car_obs'].flatten(),
                               simulator_state['action'].flatten(),
                               simulator_state['initial_conditions'],
                               # simulator_state['var_x_vel_0'].flatten(),
                               # simulator_state['var_x_vel_1'].flatten(),
                               # simulator_state['var_y_vel_0'].flatten(),
                               # simulator_state['var_y_vel_1'].flatten(),
                               # simulator_state['covar_xy_vel_0'].flatten(),
                               # simulator_state['covar_xy_vel_1'].flatten(),
                               # simulator_state['DOGMA_0'].flatten(),
                               # simulator_state['DOGMA_1'].flatten(),
                               ), axis=0)

    def restore_state(self, in_simulator_state):
        """Reset the simulation deterministically to a previously cloned state.

        This function is used in conjunction with `clone_state` for Go-Explore and Backwards Algorithm
        to do their deterministic resets.

        Parameters
        ----------
        in_simulator_state : array_like
            An array of all the simulation state variables.

        """
        # state_idx = 0
        # for var_name, var_length in self.state_vars.items():
        #     variable_val = in_simulator_state[state_idx:state_idx+var_length]
        #     if var_length == 1:
        #         variable_val = variable_val[0]
        #     setattr(self, var_name, variable_val)
        #     state_idx += var_length

        # self._peds = self._peds.reshape((self.c_num_peds, 4))
        # self._car_obs = self._car_obs.reshape((self.c_num_peds, 4))

        # Put the simulators state variables in dict form
        simulator_state = {}

        simulator_state['step'] = in_simulator_state[0]
        simulator_state['path_length'] = in_simulator_state[1]
        simulator_state['is_terminal'] = bool(in_simulator_state[2])
        simulator_state['car'] = in_simulator_state[3:7]
        simulator_state['car_accel'] = in_simulator_state[7:9]
        peds_end_index = 9 + self.c_num_peds * 4
        simulator_state['peds'] = in_simulator_state[9:peds_end_index].reshape((self.c_num_peds, 4))
        car_obs_end_index = peds_end_index + self.c_num_peds * 4
        simulator_state['car_obs'] = in_simulator_state[peds_end_index:car_obs_end_index].reshape((self.c_num_peds, 4))
        simulator_state['action'] = in_simulator_state[car_obs_end_index:car_obs_end_index + self._action.shape[0]]
        simulator_state['initial_conditions'] = in_simulator_state[car_obs_end_index + self._action.shape[0]:]

        # sim_state_idx = car_obs_end_index + self._action.shape[0] + 6
        # simulator_state['var_x_vel_0'] = in_simulator_state[sim_state_idx:sim_state_idx + self.dogma.shape[0] * self.dogma.shape[1] + 1].reshape(self.dogma.shape)
        # sim_state_idx = sim_state_idx + self.dogma.shape[0] * self.dogma.shape[1] + 1
        # simulator_state['var_x_vel_1'] = in_simulator_state[sim_state_idx:sim_state_idx + self.dogma.shape[0] * self.dogma.shape[1] + 1].reshape(self.dogma.shape)
        # sim_state_idx = sim_state_idx + self.dogma.shape[0] * self.dogma.shape[1] + 1
        # simulator_state['var_y_vel_0'] = in_simulator_state[sim_state_idx:sim_state_idx + self.dogma.shape[0] * self.dogma.shape[1] + 1].reshape(self.dogma.shape)
        # sim_state_idx = sim_state_idx + self.dogma.shape[0] * self.dogma.shape[1] + 1
        # simulator_state['var_y_vel_1'] = in_simulator_state[sim_state_idx:sim_state_idx + self.dogma.shape[0] * self.dogma.shape[1] + 1].reshape(self.dogma.shape)
        # sim_state_idx = sim_state_idx + self.dogma.shape[0] * self.dogma.shape[1] + 1
        # simulator_state['covar_xy_vel_0'] = in_simulator_state[sim_state_idx:sim_state_idx + self.dogma.shape[0] * self.dogma.shape[1] + 1].reshape(self.dogma.shape)
        # sim_state_idx = sim_state_idx + self.dogma.shape[0] * self.dogma.shape[1] + 1
        # simulator_state['covar_xy_vel_1'] = in_simulator_state[sim_state_idx:sim_state_idx + self.dogma.shape[0] * self.dogma.shape[1] + 1].reshape(self.dogma.shape)
        # sim_state_idx = sim_state_idx + self.dogma.shape[0] * self.dogma.shape[1] + 1
        # simulator_state['DOGMA_0'] = in_simulator_state[sim_state_idx:sim_state_idx + self.dogma.shape[0] * self.dogma.shape[1] * 5 + 1].reshape((5, self.dogma.shape[0], self.dogma.shape[1]))
        # sim_state_idx = sim_state_idx + self.dogma.shape[0] * self.dogma.shape[1] * 5 + 1
        # simulator_state['DOGMA_1'] = in_simulator_state[sim_state_idx:].reshape((5, self.dogma.shape[0], self.dogma.shape[1]))

        # Set ground truth of actual simulator
        self.set_ground_truth(simulator_state)

        if self.perception_type == 'dogma':
            # Save DOGMA state
            dogma_state_filename = 'step_' + str(int(simulator_state['step']))
            self.dogma.restore_state(filepath=self.dogma_state_filepath, filename=dogma_state_filename)


        # Set wrapper state variables
        self._info = []
        self.initial_conditions = np.array(simulator_state['initial_conditions'])
        self._is_terminal = simulator_state['is_terminal']
        self._path_length = simulator_state['path_length']


class DOGMA():
    def __init__(self,
                 do_plot=False,
                 shape = (128, 128),
                 p_B = 0.02,
                 Vb = 2 * 10 ** 3,
                 V = 2 * 10 ** 4,
                 state_size = 4,
                 alpha = 0.9,
                 p_A = 1.0,
                 T = 0.1,
                 p_S = 0.99,
                 scale_vel = 12.,
                 scale_acc = 2.,
                 process_pos = 0.06,
                 process_vel = 2.4,
                 process_acc = 0.2,
                 verbose = False,
                 mS = 1.,
                 epsilon = 10.,
                 epsilon_occ = 0.5,):
        # OUTPUT_DIR_IMAGES = OUTPUT_DIR + fn[0:-4] + '/'
        # if not os.path.exists(OUTPUT_DIR_IMAGES):
        #     os.makedirs(OUTPUT_DIR_IMAGES)

        self.do_plot = do_plot # Toggle me for DOGMA plots!
        self.shape = shape
        # [grids, gridglobal_x, gridglobal_y, transforms, vel_east, vel_north, acc_x, acc_y, adjust_indices] = hkl.load(
        #     DATA_DIR + fn)
        # grids = np.array(grids)
        # grids = self.crop_center(grids, self.shape[0])

        # PARAMETERS
        self.p_B = p_B  # birth probability
        # self.Vb = 2 * 10 ** 4  # number of new born particles
        # self.V = 2 * 10 ** 5  # number of consistent particles
        self.Vb = Vb  # number of new born particles
        self.V = V  # number of consistent particles
        self.state_size = state_size  # number of states: p,v: 4
        self.alpha = alpha  # information ageing (discount factor)

        self.p_A = p_A  # association probability: only relevant for Doppler measurements
        self.T = T  # measurement frequency (10 Hz)
        self.p_S = p_S  # particle persistence probability

        # velocity, acceleration variance initialization
        self.scale_vel = scale_vel
        self.scale_acc = scale_acc

        # position, velocity, acceleration process noise
        self.process_pos = process_pos
        self.process_vel = process_vel
        self.process_acc = process_acc

        # print debug values
        self.verbose = verbose

        # for plotting thresholds
        self.mS = mS  # static threshold
        self.epsilon = epsilon  # vel mag threshold
        self.epsilon_occ = epsilon_occ  # occ mag threshold
        # initialize a grid
        start = time.time()
        self.grid_cell_array = self.GridCellArray(self.shape, self.p_A)
        end = time.time()
        if self.verbose:
            print("grid_cell_array initialization took", end - start)

        # initialize a particle array
        start = time.time()
        self.particle_array = self.ParticleArray(self.V, self.grid_cell_array.get_shape(), self.state_size, self.T, self.p_S, self.scale_vel, self.scale_acc,
                                       self.process_pos, self.process_vel, self.process_acc)
        end = time.time()
        if self.verbose:
            print("particle_array initialization took", end - start)

        # # data: [N x 2 x W x D]
        # # second dimension is masses {0: m_free, 1: m_occ}
        # # in original grid: 0: unknown, 1: occupied, 2: free (raw data)
        # data = create_DST_grids(grids)

        # number of measurements in the run
        # N = data.shape[0]

        # list of 4x256x256 grids with position, velocity information
        self.DOGMA = []
        self.var_x_vel = []
        self.var_y_vel = []
        self.covar_xy_vel = []
        self.var_x_acc = []
        self.var_y_acc = []
        self.covar_xy_acc = []

    def crop_center(self, img, crop):
        if len(img.shape) == 3:
            m, x, y = img.shape
        else:
            x, y = img.shape
        startx = x // 2 - (crop // 2)
        starty = y // 2 - (crop // 2)
        if len(img.shape) == 3:
            return img[:, starty:starty + crop, startx:startx + crop]
        else:
            return img[starty:starty + crop, startx:startx + crop]

    # Populate the Dempster-Shafer measurement masses.
    def create_DST_grids(self, grids, meas_mass=0.95):

        data = []

        for i in range(grids.shape[0]):
            grid = grids[i, :, :]
            free_array = np.zeros(grid.shape)
            occ_array = np.zeros(grid.shape)

            # occupied indeces
            indeces = np.where(grid == 1)
            occ_array[indeces] = meas_mass

            # free indeces
            indeces = np.where(grid == 2)
            free_array[indeces] = meas_mass

            # car
            indeces = np.where(grid == 3)
            occ_array[indeces] = 1.

            data.append(np.stack((free_array, occ_array)))

        data = np.array(data)

        return data

    def GridCellArray(self, shape, p_A):
        return GridCellArray(shape, p_A)

    def ParticleArray(self, V, grid_cell_array_shape, state_size, T, p_S, scale_vel, scale_acc,
                                       process_pos, process_vel, process_acc, birth=False, empty_array=False):
        return ParticleArray(V, grid_cell_array_shape, state_size, T, p_S, scale_vel, scale_acc,
                                       process_pos, process_vel, process_acc, birth, empty_array)

    def MeasCellArray(self, meas_free, meas_occ, grid_cell_array_shape, pseudoG=1.):
        return MeasCellArray(meas_free, meas_occ, grid_cell_array_shape, pseudoG)

    def ParticlePrediction(self, particle_array, grid_cell_array, res=1.0):
        return ParticlePrediction(particle_array, grid_cell_array, res)

    def ParticleAssignment(self, particle_array, grid_cell_array):
        return ParticleAssignment(particle_array, grid_cell_array)

    def OccupancyPredictionUpdate(self, meas_cell_array, grid_cell_array, particle_array, p_B, alpha, check_values=None):
        if check_values is None:
            check_values = self.verbose
        return OccupancyPredictionUpdate(meas_cell_array, grid_cell_array, particle_array, p_B, alpha, check_values)

    def PersistentParticleUpdate(self, particle_array, grid_cell_array, meas_cell_array, check_values=None):
        if check_values is None:
            check_values = self.verbose
        return PersistentParticleUpdate(particle_array, grid_cell_array, meas_cell_array, check_values)

    def NewParticleInitialization(self, Vb, grid_cell_array, meas_cell_array, birth_particle_array, check_values=None):
        if check_values is None:
            check_values = self.verbose
        return NewParticleInitialization(Vb, grid_cell_array, meas_cell_array, birth_particle_array, check_values)

    def StatisticMoments(self, particle_array, grid_cell_array):
        return StatisticMoments(particle_array, grid_cell_array)

    # for now only save means of pos, vel - later can also save vel, (acc) variance, covariance
    """Need to save measurement occupancy grid instead of just particle occupancies (or in addition)!"""
    def get_dogma(self, grid_cell_array, grids, state_size, meas_grid, shape):

        ncells = grid_cell_array.get_length()

        posO = np.zeros([ncells])
        posF = np.zeros([ncells])
        velX = np.zeros([ncells])
        velY = np.zeros([ncells])
        var_x_vel = np.zeros([ncells])
        var_y_vel = np.zeros([ncells])
        covar_xy_vel = np.zeros([ncells])

        for i in range(ncells):
            posO[i] = grid_cell_array.get_cell_attr(i, "m_occ")
            posF[i] = grid_cell_array.get_cell_attr(i, "m_free")
            velX[i] = grid_cell_array.get_cell_attr(i, "mean_x_vel")
            velY[i] = grid_cell_array.get_cell_attr(i, "mean_y_vel")
            var_x_vel[i] = grid_cell_array.get_cell_attr(i, "var_x_vel")
            var_y_vel[i] = grid_cell_array.get_cell_attr(i, "var_y_vel")
            covar_xy_vel[i] = grid_cell_array.get_cell_attr(i, "covar_xy_vel")

        posO = posO.reshape(shape)
        posF = posF.reshape(shape)
        velX = velX.reshape(shape)
        velY = velY.reshape(shape)
        var_x_vel = var_x_vel.reshape(shape)
        var_y_vel = var_y_vel.reshape(shape)
        covar_xy_vel = covar_xy_vel.reshape(shape)

        newDOGMA = np.stack((posO, posF, velX, velY, meas_grid))
        return newDOGMA, var_x_vel, var_y_vel, covar_xy_vel

    def Resample(self, particle_array, birth_particle_array, particle_array_next, check_values=None):
        if check_values is None:
            check_values = self.verbose
        return Resample(particle_array, birth_particle_array, particle_array_next, check_values)

    def dogma2head_grid(self, dogma, var_x_vel, var_y_vel, covar_xy_vel, mS=4., epsilon=0.5, epsilon_occ=0.1):
        """Create heading grid for plotting tools from a DOGMA.
        USAGE:
            head_grid = dogma2head_grid(dogma, (epsilon) )
        INPUTS:
            dogma - (np.ndarray) Single DOGMA tensor (supports size of 4)
            epsilon - (opt)(float) Minimum cell vel mag required to plot heading
        OUTPUTS:
            head_grid - (np.matrix) Grid (of same shape as each vel grid) containing
                                    object headings at each cell, in rad
        """
        grid_shape = dogma[0, :, :].shape
        # Initialize grid with None's; this distinguishes from a 0rad heading!
        head_grid = np.full(grid_shape, None, dtype=float)
        vel_x, vel_y = dogma[2:4, :, :]
        m_occ, m_free = dogma[0:2, :, :]
        meas_grid = dogma[4, :, :]
        # Fill grid with heading angles where we actually have velocity
        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                # mahalanobis distance
                covar = np.array([[var_x_vel[i, j], covar_xy_vel[i, j]], [covar_xy_vel[i, j], var_y_vel[i, j]]])
                if abs(np.linalg.det(covar)) < 10 ** (-6):
                    mdist = 0.
                else:
                    mdist = np.array([vel_x[i, j], vel_y[i, j]]).dot(np.linalg.inv(covar)).dot(
                        np.array([vel_x[i, j], vel_y[i, j]]).T)
                mag = np.sqrt(vel_x[i, j] ** 2 + vel_y[i, j] ** 2)
                # occupied and with velocity
                if ((mdist > mS) and (m_occ[
                                          i, j] > epsilon_occ)):  # and (var_x_vel[i,j] < 27.) and (var_y_vel[i,j] < 27.) and (np.sqrt(vel_x[i,j]**2 + vel_y[i,j]**2) > 2.1)): # (mag > epsilon) 0.7 m/s * 3 = 2.1 0.33m/s
                    heading = np.arctan2(vel_y[i, j], vel_x[i, j])
                    head_grid[i, j] = heading
        return head_grid

    def colorwheel_plot(self, head_grid, occ_grid=None, m_occ_grid=None,
                        title="", show=True, save=False):
        return colorwheel_plot(head_grid, occ_grid, m_occ_grid,
                        title, show, save)

    def update(self, grids):
        # grids = np.array(grids)
        grids = self.crop_center(grids, self.shape[0])

        # data: [N x 2 x W x D]
        # second dimension is masses {0: m_free, 1: m_occ}
        # in original grid: 0: unknown, 1: occupied, 2: free (raw data)
        data = self.create_DST_grids(grids)

        start = time.time()

        # initializes a measurement cell array
        meas_free = data[0, 0, :, :].flatten()
        meas_occ = data[0, 1, :, :].flatten()

        meas_cell_array = self.MeasCellArray(meas_free, meas_occ, self.grid_cell_array.get_shape(), pseudoG=1.)

        # algorithm 1: ParticlePrediction (stored in particle_array)
        self.ParticlePrediction(self.particle_array, self.grid_cell_array, res=1.0)

        # algorithm 2: ParticleAssignment (stored in particle_array)
        self.ParticleAssignment(self.particle_array, self.grid_cell_array)

        # algorithm 3: OccupancyPredictionUpdate (stored in grid_cell_array)
        self.OccupancyPredictionUpdate(meas_cell_array, self.grid_cell_array, self.particle_array, self.p_B, self.alpha, check_values=self.verbose)

        # algorithm 4: PersistentParticleUpdate (stored in particle_array)
        self.PersistentParticleUpdate(self.particle_array, self.grid_cell_array, meas_cell_array, check_values=self.verbose)

        # algorithm 5: NewParticleInitialization
        if self.p_B == 0:
            empty_array = True
        else:
            empty_array = False
        birth_particle_array = self.ParticleArray(self.Vb, self.grid_cell_array.get_shape(), self.state_size, self.T, self.p_S, self.scale_vel, self.scale_acc,
                                             self.process_pos, self.process_vel, self.process_acc, birth=True, empty_array=empty_array)
        self.NewParticleInitialization(self.Vb, self.grid_cell_array, meas_cell_array, birth_particle_array, check_values=self.verbose)

        # algorithm 6: StatisticMoments (stored in grid_cell_array)
        self.StatisticMoments(self.particle_array, self.grid_cell_array)

        if self.state_size == 4:
            newDOGMA, new_var_x_vel, new_var_y_vel, new_covar_xy_vel = self.get_dogma(self.grid_cell_array, grids, self.state_size,
                                                                                 grids[0, :, :], self.shape)

            self.var_x_vel.append(new_var_x_vel)
            self.var_y_vel.append(new_var_y_vel)
            self.covar_xy_vel.append(new_covar_xy_vel)

        # save the velocities at this timestep: no real occupancy grid computed here; we will just use the measurement grid for now
        self.DOGMA.append(newDOGMA)


        # algorithm 7: Resample
        # skips particle initialization for particle_array_next because all particles will be copied in
        particle_array_next = self.ParticleArray(self.V, self.grid_cell_array.get_shape(), self.state_size, self.T, self.p_S, \
                                            self.scale_vel, self.scale_acc, self.process_pos, self.process_vel, self.process_acc,
                                            empty_array=True)
        self.Resample(self.particle_array, birth_particle_array, particle_array_next, check_values=self.verbose)
        # switch to new particle array
        self.particle_array = particle_array_next
        particle_array_next = None

        end = time.time()
        if self.verbose:
            print("Time per iteration: ", end - start)

        # Plotting: The environment is stored in grids[i] (matrix of  values (0,1,2))
        #           The DOGMA is stored in DOGMA[i]
        if (self.do_plot):
            head_grid = self.dogma2head_grid(self.DOGMA[-1], self.var_x_vel[-1], self.var_y_vel[-1], self.covar_xy_vel[-1], self.mS, self.epsilon, self.epsilon_occ)
            occ_grid = grids[0, :, :]
            title = "DOGMa Iteration"
            # pdb.set_trace()
            self.colorwheel_plot(head_grid, occ_grid=occ_grid, m_occ_grid=self.DOGMA[-1][0, :, :],
                            title=os.path.join(OUTPUT_DIR_IMAGES, title), show=True, save=True)


        # if (((i + 1) % 50 == 0) or (i == N - 1)):
        #     hkl.dump([self.DOGMA, self.var_x_vel, self.var_y_vel, self.covar_xy_vel], os.path.join(OUTPUT_DIR, fn), mode='w')
        #
        #     print
        #     "DOGMA written to hickle file."

        if len(self.DOGMA) > 1:
            # Delete dogma from more than one timestep ago
            self.DOGMA.pop(0)

        if len(self.var_x_vel) > 1:
            self.var_x_vel.pop(0)

        if len(self.var_y_vel) > 1:
            self.var_y_vel.pop(0)

        if len(self.covar_xy_vel) > 1:
            self.covar_xy_vel.pop(0)

        if len(self.var_x_acc) > 1:
            self.var_x_acc.pop(0)

        if len(self.var_y_acc) > 1:
            self.var_y_acc.pop(0)

        if len(self.covar_xy_acc) > 1:
            self.covar_xy_acc.pop(0)

        if self.verbose:
            print("Iteration complete")

        # import pdb
        # pdb.set_trace()

    def lidar_to_grid(self, pose, lidar, grid_position_m, beta, alpha):

        dx = grid_position_m.copy()  # A tensor of coordinates of all cells
        dx[0, :, :] -= pose[0]  # A matrix of all the x coordinates of the cell
        dx[1, :, :] -= pose[1]  # A matrix of all the y coordinates of the cell
        theta_to_grid = np.arctan2(dx[1, :, :], dx[0, :, :]) - pose[2]  # matrix of all bearings from robot to cell

        # Wrap to +pi / - pi
        theta_to_grid[theta_to_grid > np.pi] -= 2. * np.pi
        theta_to_grid[theta_to_grid < -np.pi] += 2. * np.pi

        dist_to_grid = scipy.linalg.norm(dx, axis=0)  # matrix of L2 distance to all cells from robot

        # Set grid to all zeros (all unknown)
        grid = np.zeros(self.shape)

        # car_x_r = map[0,:,:][np.where((np.abs(dx[0,:,:]) == np.min(np.abs(dx[0,:,:]))) & (np.abs(dx[1,:,:]) == np.min(np.abs(dx[1,:,:]))))]
        # car_y_r = map[1,:,:][np.where((np.abs(dx[0,:,:]) == np.min(np.abs(dx[0,:,:]))) & (np.abs(dx[1,:,:]) == np.min(np.abs(dx[1,:,:]))))]
        # For each laser beam
        for beam in lidar:
            r = beam[0]  # range measured
            b = beam[1]  # bearing measured

            # Calculate which cells are measured free or occupied, so we know which cells to update
            # Doing it this way is like a billion times faster than looping through each cell (because vectorized numpy is the only way to numpy)
            free_mask = (np.abs(theta_to_grid - b) <= beta / 2.0) & (dist_to_grid < (r - alpha / 2.0))
            occ_mask = (np.abs(theta_to_grid - b) <= beta / 2.0) & (np.abs(dist_to_grid - r) <= alpha / 2.0)
            car_mask = (np.abs(dx[0, :, :]) <= 2.5/2) & (np.abs(dx[1, :, :]) <= 1.4/2)

            # Set grid squares known occupied to 1
            grid[occ_mask] = 1
            #    Set grid squares known free to 2
            grid[free_mask] = 2
            #    Set grid squares known car to 3
            grid[car_mask] = 2

        return grid

    def clone_state(self, filepath, filename):
        state_dict = {'DOGMA': self.DOGMA,
                      'var_x_vel': self.var_x_vel,
                      'var_y_vel': self.var_y_vel,
                      'covar_xy_vel': self.covar_xy_vel,
                      'grid_cell_array': self.grid_cell_array,
                      'particle_array': self.particle_array,
                      }
        hkl.dump(state_dict, filepath.joinpath(filename + '.hkl'), mode='w')

    def restore_state(self, filepath, filename):
        state_dict = hkl.load(filepath.joinpath(filename + '.hkl'))

        self.DOGMA = state_dict['DOGMA']
        self.var_x_vel = state_dict['var_x_vel']
        self.var_y_vel = state_dict['var_y_vel']
        self.covar_xy_vel = state_dict['covar_xy_vel']
        self.grid_cell_array = state_dict['grid_cell_array']
        self.particle_array = state_dict['particle_array']



def create_map(xsize, ysize, grid_size, xshift=0.0, yshift=0.0, beta=10.0*np.pi/180.0):
    xsize = xsize+2 # Add extra cells for the borders
    ysize = ysize+2
    grid_size = grid_size # save this off for future use

    alpha = 1.0 # The assumed thickness of obstacles
    beta = beta # The assumed width of the laser beam
    z_max = 150.0 # The max reading from the laser

    # Pre-allocate the x and y positions of all grid positions into a 3D tensor
    # (pre-allocation = faster)
    grid_position_m = np.array([xshift + np.tile(np.arange(0, xsize*grid_size, grid_size)[:,None], (1, ysize)),
                                     yshift + np.tile(np.arange(0, ysize*grid_size, grid_size)[:,None].T, (xsize, 1))])

    return grid_position_m
