import argparse


def get_common_configs():
    parser = argparse.ArgumentParser("Choose your desired parameters")
    parser.add_argument("--agent_name", type=str, default="IQN", help="Distributional method name")
    parser.add_argument("--env_name", default="PongNoFrameskip-v4", type=str, help="Name of the environment.")
    parser.add_argument("--mem_size", default=330000, type=int, help="The memory size.")
    parser.add_argument("--lr", default=5e-5, type=float, help="The learning rate.")
    parser.add_argument("--batch_size", default=32, type=int, help="The batch size.")
    parser.add_argument("--seed", default=132, type=int, help="The random seed.")
    parser.add_argument("--interval", default=10, type=int,
                        help="The interval specifies how often different parameters should be saved and printed,"
                             " counted by episodes.")
    parser.add_argument("--do_test", action="store_true",
                        help="The flag determines whether to train the agent or play with it.")
    parser_params = parser.parse_args()

    common_params = {"state_shape": (4, 84, 84),
                     "gamma": 0.99,
                     "train_interval": 4,
                     "target_update_freq": 8000,
                     "init_mem_size_to_train": 1000,
                     "max_episodes": int(1e+4)
                     }
    # endregion
    total_params = {**vars(parser_params), **common_params}
    return total_params
