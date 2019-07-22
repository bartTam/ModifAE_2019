import argparse
import os
import os.path

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, default='CelebA')
parser.add_argument('--data_csv_path', type=str, default=None, help='Path to csv with trait ratings')
parser.add_argument('--batch_size', type=int, default=120)
parser.add_argument('--worker_number', type=int, default=4, help='The number of workers for dataset')
parser.add_argument('--input_image_size', type=int, default=128)
parser.add_argument('--log_path', type=str, default='log')
parser.add_argument('--conv_dim', type=str, default=16)
parser.add_argument('--repeat_num', type=int, default=4)
parser.add_argument('--continue_training', action='store_true')
parser.add_argument('--targets', type=str, default='Smiling,Male')
parser.add_argument('--number_of_epochs', type=int, default=100)

config = parser.parse_args()

# Split the targets into a list
config.targets = config.targets.split(',')
config.number_of_targets = len(config.targets)

# The default data_csv_path is in image_path
if config.data_csv_path is None:
    config.data_csv_path = os.path.join(config.image_path, "original_celeb.csv")

# Make sure the log directory exists
if not os.path.exists(config.log_path):
    os.mkdir(config.log_path)

# Get the last epoch that was completed if continuing
if config.continue_training:
    config.current_epoch = 0
    for dirpath, dirname, filename in os.walk(config.log_path):
        if "_T.png" in filename:
            file_epoch = int(filename.split("_T.png")[0])
            if file_epoch > config.current_epoch:
                config.current_epoch = file_epoch
