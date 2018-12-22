import argparse
import numpy as np

def get_psmcts_parser(log_dir='./'):
	parser = argparse.ArgumentParser()
	parser.add_argument('--exp_name', type=str, default="cartpole")
	parser.add_argument('--n_trial', type=int, default=5)
	parser.add_argument('--n_itr', type=int, default=10000)
	parser.add_argument('--batch_size', type=int, default=1000)
	parser.add_argument('--snapshot_mode', type=str, default="gap")
	parser.add_argument('--snapshot_gap', type=int, default=10)
	parser.add_argument('--log_dir', type=str, default=log_dir)
	parser.add_argument('--step_size', type=float, default=0.01)
	parser.add_argument('--step_size_anneal', type=float, default=1.0)
	parser.add_argument('--args_data', type=str, default=None)
	parser.add_argument('--fit_f',type=str, default="mean")
	parser.add_argument('--ec',type=float, default=1.414)
	parser.add_argument('--k',type=float, default=0.5)
	parser.add_argument('--alpha',type=float, default=0.5)
	parser.add_argument('--n_ca',type=int, default=4)
	parser.add_argument('--initial_pop',type=int, default=0)
	parser.add_argument('--log_interval', type=int, default=4000)
	parser.add_argument('--plot_tree', type=bool, default=False)
	parser.add_argument('--f_Q', type=str, default='max')
	args = parser.parse_args()
	if args.step_size_anneal == 1.0:
		args.log_dir += ('Step'+str(args.step_size)+'Ec'+str(args.ec)+'K'+str(args.k)+'A'+str(args.alpha))
	else:
		args.log_dir += ('Step'+str(args.step_size)+'Anneal'+str(args.step_size_anneal)+'Ec'+str(args.ec)+'K'+str(args.k)+'A'+str(args.alpha))
	if args.initial_pop > 0:
		args.log_dir += ('InitP'+str(args.initial_pop))
	args.log_dir += 'Q'+args.f_Q
	return args