import pickle
import numpy as np
import pdb

# from examples.AV.example_runner_ba_av import runner as ba_runner
# from examples.AV.example_runner_drl_av import runner as drl_runner
# from examples.hifi.runners.runner_ba_av import runner as ba_runner
from examples.hifi.runners.runner_drl_av import runner as drl_runner
from examples.hifi.runners.perception_runner_drl_av import runner as perception_drl_runner
from examples.hifi.runners.perception_runner_ba import runner as perception_ba_runner

from examples.AV.example_runner_ge_av import runner as go_explore_runner
from examples.AV.example_runner_mcts_av import runner as mcts_runner

from examples.hifi.simulators.mixed_fidelity_av_simuator import MixedFidelityAVSimulator
from examples.hifi.rewards.perception_av_reward import PerceptionAVReward
#
# ---------------------------------------  -----------------
# AverageDiscountedReturn                    -8383.93
# AverageReturn                             -13581.9
# Entropy                                        3.87807
# Extras/EpisodeRewardMean                   -3629.9
# Iteration                                    215
# LinearFeatureBaseline/ExplainedVariance       -0.000272292
# MaxReturn                                   -281.941
# MinReturn                                -175937
# NumTrajs                                      25
# Perplexity                                    48.3306
# StdReturn                                  45087.1
# lstm_policy/Entropy                            3.64514
# lstm_policy/KL                                 1.57999
# lstm_policy/KLBefore                           1.57999
# lstm_policy/LossAfter                          0.0902296
# lstm_policy/LossBefore                         0.0902296
# lstm_policy/dLoss                              0
# ---------------------------------------  -----------------
# Backward Results -- Expert Trajectory Reward:  -286.4664442296516  -- Best Reward at step  23 :  -259.49155  -- Best Final Reward:  -281.94098


def perception_lofi_to_hifi_expert_trajectory(lofi_expert_trajectory, s_0, sim_args, reward_args):
    hifi_expert_trajectory = []
    sim_args['open_loop'] = False
    sim = MixedFidelityAVSimulator(**sim_args)
    reward_function = PerceptionAVReward(**reward_args)
    sim.reset(s_0=s_0)
    for traj_step in lofi_expert_trajectory:
        # pdb.set_trace()
        traj_step_dict = {}

        action = traj_step['action'][0:3]
        obs = sim.step(action=action)

        traj_step_dict['action'] = action
        traj_step_dict['observation'] = obs
        traj_step_dict['reward'] = reward_function.give_reward(action=action,
                                                               info=sim.get_reward_info())
        traj_step_dict['state'] = np.concatenate((sim.clone_state(), traj_step['state'][-2:]))

        hifi_expert_trajectory.append(traj_step_dict)

    return hifi_expert_trajectory

if __name__ == '__main__':
    # Which algorithms to run
    RUN_LOFI = False
    RUN_BA = True
    RUN_HIFI = False

    lofi_use_tracker = True
    hifi_use_tracker = True

    blackbox_sim_state = True
    open_loop = True
    fixed_init_state = True

    n_parallel = 16

    # Overall settings
    max_path_length = 50
    s_0 = [0.0, -4.0, 1.0, 11.17, -35.0]
    base_log_dir = './data'
    dogma_state_filepath = base_log_dir + '/dogma/expert_trajectory'
    # experiment settings
    run_experiment_args = {'snapshot_mode': 'last',
                           'snapshot_gap': 1,
                           'log_dir': None,
                           'exp_name': None,
                           'seed': 0,
                           'n_parallel': n_parallel,
                           'tabular_log_file': 'progress.csv'
                           }

    # runner settings
    runner_args = {'n_epochs': 100,
                   'batch_size': 500,
                   'plot': False,
                   'store_paths': True,
                   }

    # env settings
    env_args = {'id': 'ast_toolbox:GoExploreAST-v1',
                'blackbox_sim_state': blackbox_sim_state,
                'open_loop': open_loop,
                'fixed_init_state': fixed_init_state,
                's_0': s_0,
                }



    # reward settings
    reward_args = {'use_heuristic': True}

    # spaces settings
    spaces_args = {}


    #####################################################
    # Run Lofi
    #####################################################
    if RUN_LOFI:

        # simulation settings
        from examples.hifi.simulators.perception_av_simulator import PerceptionAVSimulator
        sub_sim_args = {'use_tracker':lofi_use_tracker,
                        'noise_vector_size': 6}
        sim_args = {'blackbox_sim_state': blackbox_sim_state,
                    'open_loop': open_loop,
                    'fixed_initial_state': fixed_init_state,
                    'max_path_length': max_path_length,
                    'simulator_cls':PerceptionAVSimulator,
                    'simulator_args':sub_sim_args,
                    }

        # DRL Settings

        drl_policy_args = {'name': 'lstm_policy',
                           'hidden_dim': 64,
                           }

        drl_baseline_args = {}

        drl_algo_args = {'max_path_length': max_path_length,
                         'discount': 0.99,
                         'lr_clip_range': 1.0,
                         'max_kl_step': 1.0,
                         # 'log_dir':None,
                         }

        run_experiment_args['log_dir'] = base_log_dir + '/lofi'
        run_experiment_args['exp_name'] = 'drl'

        drl_algo_args['max_path_length'] = max_path_length

        # Run DRL
        drl_runner(
            env_args=env_args,
            run_experiment_args=run_experiment_args,
            sim_args=sim_args,
            reward_args=reward_args,
            spaces_args=spaces_args,
            policy_args=drl_policy_args,
            baseline_args=drl_baseline_args,
            algo_args=drl_algo_args,
            runner_args=runner_args,
            save_expert_trajectory=True,
        )

        with open(base_log_dir + '/lofi' + '/expert_trajectory.pkl', 'rb') as f:
            expert_trajectories = pickle.load(f)

        hifi_expert_trajectory = perception_lofi_to_hifi_expert_trajectory(lofi_expert_trajectory=expert_trajectories,
                                                                           s_0=s_0,
                                                                           sim_args=sim_args,
                                                                           reward_args=reward_args)

        with open(base_log_dir + '/hifi' + '/expert_trajectory.pkl', 'wb') as f:
            pickle.dump(hifi_expert_trajectory, f)

    #####################################################
    # Run BA Hifi
    #####################################################
    if RUN_BA:
        # simulation settings
        from examples.hifi.simulators.perception_av_simulator import PerceptionAVSimulator

        perception_args = {'do_plot': False,
                           'shape': (128, 128),
                           'p_B': 0.02,
                           'Vb': 2 * 10 ** 3,
                           'V': 2 * 10 ** 4,
                           'state_size': 4,
                           'alpha': 0.9,
                           'p_A': 1.0,
                           'T': 0.1,
                           'p_S': 0.99,
                           'scale_vel': 12.0,
                           'scale_acc': 2.0,
                           'process_pos': 0.06,
                           'process_vel': 2.4,
                           'process_acc': 0.2,
                           'verbose': False,
                           'mS': 1.0,
                           'epsilon': 10.0,
                           'epsilon_occ': 0.5,
                           'num_beams': 30,
                           'total_beam_width': np.pi,
                           # 'beam_width': 0,
                           'target_thickness': 1.0,
                           'grid_size': 0.5,
                           'xshift': -35,
                           'yshift': -6.5,
                           'dogma_state_filepath': dogma_state_filepath,
                           }
        sub_sim_args = {'use_tracker': hifi_use_tracker,
                        'perception_type': 'dogma',
                        'perception_args': perception_args,
                        'noise_vector_size': 3}

        sim_args = {'blackbox_sim_state': blackbox_sim_state,
                    'open_loop': open_loop,
                    'fixed_initial_state': fixed_init_state,
                    'max_path_length': max_path_length,
                    'simulator_cls':PerceptionAVSimulator,
                    'simulator_args':sub_sim_args,
                    }
        # BA Settings
        ba_algo_args = {'expert_trajectory': None,
                        'max_path_length': max_path_length,
                        'epochs_per_step': 10,
                        'scope': None,
                        'discount': 0.99,
                        'gae_lambda': 1.0,
                        'center_adv': True,
                        'positive_adv': False,
                        'fixed_horizon': False,
                        'pg_loss': 'surrogate_clip',
                        'lr_clip_range': 1.0,
                        'max_kl_step': 1.0,
                        'policy_ent_coeff': 0.0,
                        'use_softplus_entropy': False,
                        'use_neg_logli_entropy': False,
                        'stop_entropy_gradient': False,
                        'entropy_method': 'no_entropy',
                        'name': 'PPO',
                        }

        ba_baseline_args = {}

        ba_policy_args = {'name': 'lstm_policy',
                          'hidden_dim': 64,
                          }

        spaces_args = {'observation_low': np.array(s_0),
                       'observation_high': np.array(s_0),
                       'action_low': np.array([-1.0,-1.0,-3.0]),
                       'action_high': np.array([1.0,1.0,3.0]),}

        sampler_args = {"open_loop": open_loop,
                        'n_envs': n_parallel}

        # with open(base_log_dir + '/lofi' + '/expert_trajectory.pkl', 'rb') as f:
        #     expert_trajectories = pickle.load(f)
        #
        # hifi_expert_trajectory = perception_lofi_to_hifi_expert_trajectory(lofi_expert_trajectory=expert_trajectories,
        #                                                                    s_0=s_0,
        #                                                                    sim_args=sim_args,
        #                                                                    reward_args=reward_args)
        #
        # with open(base_log_dir + '/hifi' + '/expert_trajectory.pkl', 'wb') as f:
        #     pickle.dump(hifi_expert_trajectory, f)


        with open(base_log_dir + '/hifi' + '/expert_trajectory.pkl', 'rb') as f:
            hifi_expert_trajectory = pickle.load(f)


        # pdb.set_trace()
        run_experiment_args['log_dir'] = base_log_dir + '/ba'
        ba_algo_args['expert_trajectory'] = hifi_expert_trajectory

        perception_ba_runner(
            env_args=env_args,
            run_experiment_args=run_experiment_args,
            sim_args=sim_args,
            reward_args=reward_args,
            spaces_args=spaces_args,
            policy_args=ba_policy_args,
            baseline_args=ba_baseline_args,
            algo_args=ba_algo_args,
            runner_args=runner_args,
            sampler_args=sampler_args,
        )

    #####################################################
    # Run Hifi Baseline
    #####################################################
    if RUN_HIFI:
        # simulation settings
        from examples.hifi.simulators.perception_av_simulator import PerceptionAVSimulator
        perception_args = {'do_plot': False,
                           'shape': (128, 128),
                           'p_B': 0.02,
                           'Vb': 2 * 10 ** 2,
                           'V': 2 * 10 ** 3,
                           'state_size': 4,
                           'alpha': 0.9,
                           'p_A': 1.0,
                           'T': 0.1,
                           'p_S': 0.99,
                           'scale_vel': 12.0,
                           'scale_acc': 2.0,
                           'process_pos': 0.06,
                           'process_vel': 2.4,
                           'process_acc': 0.2,
                           'verbose': False,
                           'mS': 1.0,
                           'epsilon': 10.0,
                           'epsilon_occ': 0.5,
                           'num_beams': 30,
                           'total_beam_width': np.pi,
                           # 'beam_width': 0,
                           'target_thickness': 1.0,
                           'grid_size': 0.5,
                           'xshift': -35,
                           'yshift': -6.5,
                           'dogma_state_filepath': dogma_state_filepath
                           }
        sub_sim_args = {'use_tracker': hifi_use_tracker,
                        'perception_type': 'dogma',
                        'perception_args': perception_args,
                        'noise_vector_size': 3}

        sim_args = {'blackbox_sim_state': blackbox_sim_state,
                    'open_loop': open_loop,
                    'fixed_initial_state': fixed_init_state,
                    'max_path_length': max_path_length,
                    'simulator_cls':PerceptionAVSimulator,
                    'simulator_args':sub_sim_args,
                    }

        spaces_args = {'observation_low': np.array(s_0),
                       'observation_high': np.array(s_0),
                       'action_low': np.array([-1.0,-1.0,-3.0]),
                       'action_high': np.array([1.0,1.0,3.0]),}

        # DRL Settings

        drl_policy_args = {'name': 'lstm_policy',
                           'hidden_dim': 64,
                           }

        drl_baseline_args = {}

        drl_algo_args = {'max_path_length': max_path_length,
                         'discount': 0.99,
                         'lr_clip_range': 1.0,
                         'max_kl_step': 1.0,
                         # 'log_dir':None,
                         }

        run_experiment_args['log_dir'] = base_log_dir + '/hifi'
        run_experiment_args['exp_name'] = 'drl'

        drl_algo_args['max_path_length'] = max_path_length

        # Run DRL
        perception_drl_runner(
            env_args=env_args,
            run_experiment_args=run_experiment_args,
            sim_args=sim_args,
            reward_args=reward_args,
            spaces_args=spaces_args,
            policy_args=drl_policy_args,
            baseline_args=drl_baseline_args,
            algo_args=drl_algo_args,
            runner_args=runner_args,
            save_expert_trajectory=False,
        )


