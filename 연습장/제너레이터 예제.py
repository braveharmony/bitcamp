import os
import numpy as np
from tensorflow.keras.utils import Sequence

class NumpyDataGenerator(Sequence):
    def __init__(self, data_dir, target_dir, batch_size):
        self.data_dir = data_dir
        self.target_dir = target_dir
        self.batch_size = batch_size
        self.file_list = sorted(os.listdir(data_dir))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        data_path = os.path.join(self.data_dir, self.file_list[index])
        target_path = os.path.join(self.target_dir, self.file_list[index])

        data = np.load(data_path)
        target = np.load(target_path)

        return data, target
    
    