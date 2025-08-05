import os

from douzero.dmc import parser, train

if __name__ == '__main__':
    flags = parser.parse_args()
    flags.num_actors = 8
    flags.num_threads = 32
    flags.load_model = False
    flags.batch_size = 32
    flags.sleep_time = 20
    flags.savedir = "mini_oracle_batch16"
    flags.use_wandb = True
    flags.save_interval = 20
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    train(flags)
