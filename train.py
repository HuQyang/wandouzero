import os

from douzero.dmc import parser, train

if __name__ == '__main__':
    flags = parser.parse_args()
    flags.num_actors = 12
    flags.num_threads = 4
    flags.load_model = False
    flags.batch_size = 128
    flags.sleep_time = 20
    flags.savedir = "mini_oracle_batch16"
    flags.use_wandb = False
    flags.save_interval = 20
    flags.unroll_length = 60
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    train(flags)
