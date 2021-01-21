# -*- coding: utf-8 -*-
"""
Created on Fri Nov 03 09:40:51 2017

@author: Masha

Main function for the particle filter algorithm. Handles conversion of data types.

D. Nuss, S. Reuter, M. Thom, T. Yuan, G. Krehl, M. Maile, A. Gern, and K. Dietmayer. A
random finite set approach for dynamic occupancy grid maps with real-time application. arXiv,
abs/1605.02406, 2016.

NOTE: velocities are in res/s -> res = 0.33 m

"""
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pdb
from scipy.stats import itemfreq

seed = 1987
np.random.seed(seed)

from NewParticleInitialization import *
from ParticlePrediction import *
from ParticleAssignment import *
from OccupancyPredictionUpdate import *
from PersistentParticleUpdate import *
from StatisticMoments import *
from Resample import *

from Particle import *
from Grid import *

import pickle
import hickle as hkl
import time
import sys
import os
import math
sys.path.insert(0, '..')
from PlotTools import colorwheel_plot

DATA_DIR = '../../Data/SensorMeasurements/'
OUTPUT_DIR = "../../Data/ParticleFilter/VelocityGrids/"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def crop_center(img,crop):
    m,x,y = img.shape
    startx = x//2-(crop//2)
    starty = y//2-(crop//2)
    return img[:,starty:starty+crop,startx:startx+crop]

# Populate the Dempster-Shafer measurement masses.
def create_DST_grids(grids, meas_mass=0.95):

    data = []

    for i in range(grids.shape[0]):

        grid = grids[i,:,:]
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

def main():

    for fn in sorted(os.listdir(DATA_DIR)):
        print fn

        if (fn[-3:] == 'hkl'):

            OUTPUT_DIR_IMAGES = OUTPUT_DIR + fn[0:-4] + '/'
            if not os.path.exists(OUTPUT_DIR_IMAGES):
                os.makedirs(OUTPUT_DIR_IMAGES)

            shape = (128,128)
            [grids, gridglobal_x, gridglobal_y, transforms, vel_east, vel_north, acc_x, acc_y, adjust_indices] = hkl.load(DATA_DIR + fn)
            grids = np.array(grids)
            grids = crop_center(grids, shape[0])

            do_plot = True # Toggle me for DOGMA plots!

            # PARAMETERS
            p_B = 0.02                                            # birth probability
            Vb = 2*10**4                                          # number of new born particles
            V = 2*10**5                                           # number of consistent particles
            state_size = 4                                        # number of states: p,v: 4
            alpha = 0.9                                           # information ageing (discount factor)

            p_A = 1.0                                             # association probability: only relevant for Doppler measurements
            T = 0.1                                               # measurement frequency (10 Hz)
            p_S = 0.99                                            # particle persistence probability

            # velocity, acceleration variance initialization
            scale_vel = 12.
            scale_acc = 2.

            # position, velocity, acceleration process noise
            process_pos = 0.06
            process_vel = 2.4
            process_acc = 0.2

            # print debug values
            verbose = False

            # for plotting thresholds
            mS = 3. 			# static threshold
            epsilon = 10.   	# vel mag threshold
            epsilon_occ = 0.75 	# occ mag threshold

            # initialize a grid
            start = time.time()
            grid_cell_array = GridCellArray(shape, p_A)
            end =  time.time()
            print "grid_cell_array initialization took", end - start

            # initialize a particle array
            start = time.time()
            particle_array = ParticleArray(V, grid_cell_array.get_shape(), state_size, T, p_S, scale_vel, scale_acc, process_pos, process_vel, process_acc)
            end =  time.time()
            print "particle_array initialization took", end - start

            # data: [N x 2 x W x D]
            # second dimension is masses {0: m_free, 1: m_occ}
            # in original grid: 0: unknown, 1: occupied, 2: free (raw data)
            data = create_DST_grids(grids)

            # number of measurements in the run
            N = data.shape[0]

            # list of 4x256x256 grids with position, velocity information
            DOGMA = []
            var_x_vel = []
            var_y_vel = []
            covar_xy_vel = []
            var_x_acc = []
            var_y_acc = []
            covar_xy_acc = []

            # run particle filter iterations
            for i in range(N):

                start = time.time()

                # initializes a measurement cell array
                meas_free = data[i,0,:,:].flatten()
                meas_occ = data[i,1,:,:].flatten()

                meas_cell_array = MeasCellArray(meas_free, meas_occ, grid_cell_array.get_shape(), pseudoG = 1.)

                # algorithm 1: ParticlePrediction (stored in particle_array)
                ParticlePrediction(particle_array, grid_cell_array, res=1.0)

                # algorithm 2: ParticleAssignment (stored in particle_array)
                ParticleAssignment(particle_array, grid_cell_array)

                # algorithm 3: OccupancyPredictionUpdate (stored in grid_cell_array)
                OccupancyPredictionUpdate(meas_cell_array, grid_cell_array, particle_array, p_B, alpha, check_values = verbose)

                # algorithm 4: PersistentParticleUpdate (stored in particle_array)
                PersistentParticleUpdate(particle_array, grid_cell_array, meas_cell_array, check_values = verbose)

                # algorithm 5: NewParticleInitialization
                if p_B == 0:
                    empty_array = True
                else:
                    empty_array = False
                birth_particle_array = ParticleArray(Vb, grid_cell_array.get_shape(), state_size, T, p_S, scale_vel, scale_acc, process_pos, process_vel, process_acc, birth = True, empty_array = empty_array)
                NewParticleInitialization(Vb, grid_cell_array, meas_cell_array, birth_particle_array, check_values = verbose)

                # algorithm 6: StatisticMoments (stored in grid_cell_array)
                StatisticMoments(particle_array, grid_cell_array)

                if state_size == 4:

                    newDOGMA, new_var_x_vel, new_var_y_vel, new_covar_xy_vel = get_dogma(grid_cell_array, grids, state_size, grids[i,:,:], shape)

                    var_x_vel.append(new_var_x_vel)
                    var_y_vel.append(new_var_y_vel)
                    covar_xy_vel.append(new_covar_xy_vel)

                # save the velocities at this timestep: no real occupancy grid computed here; we will just use the measurement grid for now
                DOGMA.append(newDOGMA)

                # algorithm 7: Resample
                # skips particle initialization for particle_array_next because all particles will be copied in
                particle_array_next = ParticleArray(V, grid_cell_array.get_shape(), state_size, T, p_S, \
                                                scale_vel, scale_acc, process_pos, process_vel, process_acc, empty_array = True)
                Resample(particle_array, birth_particle_array, particle_array_next, check_values = verbose)
                # switch to new particle array
                particle_array = particle_array_next
                particle_array_next = None

                end = time.time()
                print "Time per iteration: ", end - start

                # Plotting: The environment is stored in grids[i] (matrix of  values (0,1,2))
                #           The DOGMA is stored in DOGMA[i]
                if (do_plot):
                    head_grid = dogma2head_grid(DOGMA[i], var_x_vel[i], var_y_vel[i], covar_xy_vel[i], mS, epsilon, epsilon_occ)
                    occ_grid = grids[i,:,:]
                    title = "DOGMa Iteration %d" % i
                    colorwheel_plot(head_grid, occ_grid=occ_grid, m_occ_grid = DOGMA[i][0,:,:], title=os.path.join(OUTPUT_DIR_IMAGES, title), show=True, save=True)

                if (((i+1)%50 == 0) or (i == N-1)):

                    hkl.dump([DOGMA,var_x_vel, var_y_vel, covar_xy_vel], os.path.join(OUTPUT_DIR, fn), mode='w')

                    print "DOGMA written to hickle file."

                print "Iteration ", i, " complete"

    return

# for now only save means of pos, vel - later can also save vel, (acc) variance, covariance
"""Need to save measurement occupancy grid instead of just particle occupancies (or in addition)!"""
def get_dogma(grid_cell_array, grids, state_size, meas_grid, shape):

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

    newDOGMA = np.stack((posO,posF,velX,velY,meas_grid))
    return newDOGMA, var_x_vel, var_y_vel, covar_xy_vel

def dogma2head_grid(dogma, var_x_vel, var_y_vel, covar_xy_vel, mS = 4., epsilon=0.5, epsilon_occ=0.1):
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
    grid_shape = dogma[0,:,:].shape
    # Initialize grid with None's; this distinguishes from a 0rad heading!
    head_grid = np.full(grid_shape, None, dtype=float)
    vel_x, vel_y = dogma[2:4,:,:]
    m_occ, m_free = dogma[0:2,:,:]
    meas_grid = dogma[4,:,:]
    # Fill grid with heading angles where we actually have velocity
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            # mahalanobis distance
            covar = np.array([[var_x_vel[i,j], covar_xy_vel[i,j]], [covar_xy_vel[i,j], var_y_vel[i,j]]])
            if abs(np.linalg.det(covar)) < 10**(-6):
                mdist = 0.
            else:
                mdist = np.array([vel_x[i,j], vel_y[i,j]]).dot(np.linalg.inv(covar)).dot(np.array([vel_x[i,j], vel_y[i,j]]).T)
            mag = np.sqrt(vel_x[i,j]**2 + vel_y[i,j]**2)
            # occupied and with velocity
            if ((mdist > mS) and (m_occ[i,j] > epsilon_occ)): # and (var_x_vel[i,j] < 27.) and (var_y_vel[i,j] < 27.) and (np.sqrt(vel_x[i,j]**2 + vel_y[i,j]**2) > 2.1)): # (mag > epsilon) 0.7 m/s * 3 = 2.1 0.33m/s
                heading = np.arctan2(vel_y[i,j], vel_x[i,j])
                head_grid[i,j] = heading
    return head_grid

if __name__ == "__main__":
    main()
