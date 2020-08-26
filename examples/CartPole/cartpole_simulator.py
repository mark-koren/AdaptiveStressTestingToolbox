from CartPole.cartpole import CartPoleEnv

from src.ast_toolbox.simulators.nn_sut_simulator import NNSUTSimulator


class CartpoleSimulator(NNSUTSimulator):
    """
    neural network system under test simulator
    """

    def __init__(self,
                 use_seed=False,
                 nd=1,
                 **kwargs):

        # initialize the base Simulator
        super().__init__(**kwargs, env=CartPoleEnv(use_seed=use_seed, nd=nd))
