from tensorflow.keras.applications import VGG16,ResNet50V2
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Conv2D,Dropout,Flatten,Input,GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import numpy as np

(x_train,y_train),(x_test,y_test)=cifar10.load_data()
x_train=x_train/255.
x_test=x_test/255.

base_model=VGG16(include_top=False)
x=base_model.output
print(x)
x=GlobalAveragePooling2D()(x)
output1=Dense(10,activation='softmax')(x)

model=Model(inputs=base_model.input,outputs=output1)

es=EarlyStopping(monitor='val_acc',mode='max',patience=5,verbose=0,restore_best_weights=True)

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics='acc')
model.fit(x_train,y_train,epochs=100,batch_size=100,validation_data=(x_test,y_test),callbacks=es)
acc_no_trained=accuracy_score(y_test,np.argmax(model.predict(x_test),axis=1))

print(f"acc with no trained : {acc_no_trained}")





from io import StringIO
import sys
buffer = StringIO()
sys.stdout = buffer
model.summary()
summary = buffer.getvalue()
sys.stdout = sys.__stdout__

print('===================')
print(summary)