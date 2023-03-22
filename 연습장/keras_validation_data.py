# 0. seed initialization
import numpy as np
import tensorflow as tf
import random
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)