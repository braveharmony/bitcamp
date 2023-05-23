import numpy as np
import matplotlib.pyplot as plt
x=10
y=10
w=11
lr=0.0001
epochs=5000000

patience=5
min_loss=np.inf
count=0
for step in range(1,epochs+1):
    hypothesis=x*w
    loss=(hypothesis-y)**2
    print(f'step : {step} \tloss : {round(loss,4)} \tPredict : {round(hypothesis,4)}')
    w=w-2*lr*(hypothesis-y)*x
    
    # up_pred=x*(w+lr)
    # down_pred=x*(w-lr)
    # up_loss=(up_pred-y)**2
    # down_loss=(down_pred-y)**2
    # if up_loss<down_loss and up_loss<loss:
    #     w=w+lr
    # elif up_loss>down_loss and down_loss<loss:
    #     w=w-lr
    
    if min_loss>loss:
        count=0
        min_loss=loss
    else : count+=1
    if count==patience:
        break