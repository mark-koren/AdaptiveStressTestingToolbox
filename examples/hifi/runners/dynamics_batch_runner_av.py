import pickle

# from examples.AV.example_runner_ba_av import runner as ba_runner
# from examples.AV.example_runner_drl_av import runner as drl_runner
from examples.hifi.runners.runner_ba_av import runner as ba_runner
from examples.hifi.runners.runner_drl_av import runner as drl_runner
from examples.AV.example_runner_ge_av import runner as go_explore_runner
from examples.AV.example_runner_mcts_av import runner as mcts_runner

if __name__ == '__main__':
    # Which algorithms to run
    RUN_LOFI = False
    RUN_BA = False
    RUN_HIFI = True

    hifi_decimals = 32
    lofi_decimals = 2

    blackbox_sim_state = True
    open_loop = True
    fixed_init_state = True

    # Overall settings
    max_path_length = 50
    s_0 = [0.0, -4.0, 1.0, 11.17, -35.0]
    base_log_dir = './data'
    # experiment settings
    run_experiment_args = {'snapshot_mode': 'last',
                           'snapshot_gap': 1,
                           'log_dir': None,
                           'exp_name': None,
                           'seed': 0,
                           'n_parallel': 8,
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
    from examples.hifi.simulators.dynamics_av_simulator import DynamicsAVSimulator
    sub_sim_args = {'decimals':lofi_decimals,}
    sim_args = {'blackbox_sim_state': blackbox_sim_state,
                'open_loop': open_loop,
                'fixed_initial_state': fixed_init_state,
                'max_path_length': max_path_length,
                'simulator_cls':DynamicsAVSimulator,
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

#####################################################
# Run BA Hifi
#####################################################
if RUN_BA:
    # simulation settings
    from examples.hifi.simulators.dynamics_av_simulator import DynamicsAVSimulator
    sub_sim_args = {'decimals':hifi_decimals,}
    sim_args = {'blackbox_sim_state': blackbox_sim_state,
                'open_loop': open_loop,
                'fixed_initial_state': fixed_init_state,
                'max_path_length': max_path_length,
                'simulator_cls':DynamicsAVSimulator,
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

    with open(base_log_dir + '/lofi' + '/expert_trajectory.pkl', 'rb') as f:
        expert_trajectories = pickle.load(f)

    run_experiment_args['log_dir'] = base_log_dir + '/ba'
    ba_algo_args['expert_trajectory'] = expert_trajectories

    ba_runner(
        env_args=env_args,
        run_experiment_args=run_experiment_args,
        sim_args=sim_args,
        reward_args=reward_args,
        spaces_args=spaces_args,
        policy_args=ba_policy_args,
        baseline_args=ba_baseline_args,
        algo_args=ba_algo_args,
        runner_args=runner_args,
    )

#####################################################
# Run Hifi Baseline
#####################################################
if RUN_HIFI:
    # simulation settings
    from examples.hifi.simulators.dynamics_av_simulator import DynamicsAVSimulator
    sub_sim_args = {'decimals':hifi_decimals,}
    sim_args = {'blackbox_sim_state': blackbox_sim_state,
                'open_loop': open_loop,
                'fixed_initial_state': fixed_init_state,
                'max_path_length': max_path_length,
                'simulator_cls':DynamicsAVSimulator,
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

    run_experiment_args['log_dir'] = base_log_dir + '/hifi'
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
        save_expert_trajectory=False,
    )
