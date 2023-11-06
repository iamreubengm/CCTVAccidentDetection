import cv2
from darknet_images import *


args = parser()
check_arguments_errors(args)

random.seed(3)  # deterministic bbox colors
network, class_names, class_colors = darknet.load_network(
    args.config_file,
    args.data_file,
    args.weights,
    batch_size=args.batch_size
)
