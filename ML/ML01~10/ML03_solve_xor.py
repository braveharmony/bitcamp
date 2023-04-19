import tensorflow as tf
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron

def model_run(x,y,model_list):
    for model in model_list:
        model=model()
        model.fit(x,xor)
        print(f'{model.score(x,xor)}')
        
x1=(0,1)
x2=(0,1)
x=[[i,j]for i in x1 for j in x2]
xor=[i ^ j for i in x1 for j in x2]

models=(SVC,Perceptron)

model_run(x,xor,models)