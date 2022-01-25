# extracted eyes + keras aumentatgion + cnn

import pandas as pd
import os
from datahandler import DataHandlerEye
from MyIO import IO
import models
import numpy as np
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
from skimage import io as sio
import sklearn

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


io = IO(__file__, 'train2',inputScript='eye_detection_total_face',inputRunName='eye_detection',inputTag=1)
config = io.getRunConfig()
dataHandler = DataHandlerEye(io)
dataHandler.clearData()
dataHandler.y = pd.get_dummies(dataHandler.y['target'])
dataHandler.X = dataHandler.X/255.0
layers = {'cnn':[[64,(3,3),(2,2)],[64,(3,3),(3,3)]],
          'dense':[512]}
# layers = {'cnn':[[64,(3,3),(3,3)],[32,(3,3),(2,2)]],
#           'dense':[500]}
# layers = {'cnn':[[128,(3,3),(3,3)],],
#           'dense':[64,32]}
model = models.cnn_model2(io=io,lyrs=layers,X=dataHandler.X,y=dataHandler.y)
model.compile()
X_train, X_test, y_train, y_test = dataHandler.splitData(dataHandler.X,dataHandler.y,test_size=.2)
X_val,X_test,y_val,y_test = dataHandler.splitData(X_test,y_test,test_size=.6)
np.save(io.getOutputPath()+'X_train.npy',X_train)
np.save(io.getOutputPath()+'X_test.npy',X_test)
np.save(io.getOutputPath()+'y_train.npy',y_train)
np.save(io.getOutputPath()+'y_test.npy',y_test)

cw = sklearn.utils.compute_class_weight('balanced',[0,1,2,3],np.argmax(y_train.values,axis=1))
cw = cw
it_train = dataHandler.createDataGenerator(X_train,y_train.values)
it_val = dataHandler.createPureDataGenerator(X_val,y_val)
model.fit_generator(it_train,it_val,epochs=500,steps_per_epoch=60,cw=cw)

print('mean:'+y_train.mean().__str__())
print('mean:'+y_test.mean().__str__())
# model.train(X_train,y_train,epochs=500)
model.test(X_test,y_test)
print('mean:'+y_train.mean().__str__())
print('mean:'+y_test.mean().__str__())
print('end')
