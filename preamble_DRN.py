import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
c_path = os.path.dirname(os.path.realpath(__file__))
import logging
tf.get_logger().setLevel(logging.ERROR)
