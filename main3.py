# handly agumented data + cnn

import pandas as pd
import os
from datahandler import DataHandlerEyeAgm
from MyIO import IO
import models
import numpy as np
from skimage import io as sio
import tensorflow as tf
import sklearn
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

io = IO(__file__, 'train3',inputScript='eye_detection_agumentation',inputRunName='eye_detection',inputTag=1)
config = io.getRunConfig()
dataHandler = DataHandlerEyeAgm(io)
dataHandler.clearData()
dataHandler.y = pd.get_dummies(dataHandler.y['target'])
dataHandler.X = dataHandler.X/255.0
layers = {'cnn':[[32,(3,3),(3,3)],[64,(3,3),(3,3)]],
          'dense':[64]}
model = models.cnn_model2(io=io,lyrs=layers,X=dataHandler.X,y=dataHandler.y)
model.compile()
X_train, X_test, y_train, y_test = dataHandler.splitData(dataHandler.X,dataHandler.y)
np.save(io.getOutputPath()+'X_train.npy',X_train)
np.save(io.getOutputPath()+'X_test.npy',X_test)
np.save(io.getOutputPath()+'y_train.npy',y_train)
np.save(io.getOutputPath()+'y_test.npy',y_test)
print('mean:'+y_train.mean().__str__())
print('mean:'+y_test.mean().__str__())
cw = sklearn.utils.compute_class_weight('balanced',[0,1,2,3],np.argmax(y_train.values,axis=1))
model.train(X_train,y_train,epochs=500,cw=cw)
model.test(X_test,y_test)
print('mean:'+y_train.mean().__str__())
print('mean:'+y_test.mean().__str__())
print('end')
