import argparse

parser = argparse.ArgumentParser()

# =================== gpu ================== #
parser.add_argument('--gpu_id', type=str, default='0')

# =================== data set & path ================== #
parser.add_argument('--data_set_type', type=str, default='porto')  # geolife or porto
parser.add_argument('--origin_set', type=str, default='origin_set')
parser.add_argument('--train_num', type=int, default=3000)
parser.add_argument('--val_num', type=int, default=3000)

# =================== random seed ================== #
parser.add_argument('--seed', type=int, default=808)

# ==================== training ==================== #

# ==================== model ==================== #
parser.add_argument('--use_GCN', type=bool, default=True)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--num_layers', type=int, default=1)
