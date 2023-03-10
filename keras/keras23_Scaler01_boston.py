from sklearn.datasets import load_boston
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

datasets = load_boston()
x=datasets['data']
y=datasets['target']

print(type(x))
print(x)