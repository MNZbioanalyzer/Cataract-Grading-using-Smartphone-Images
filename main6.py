from keras.applications.vgg16 import VGG16
from keras.applications import InceptionV3
from keras.applications import MobileNet
from keras.applications import ResNet101
from keras.preprocessing import image
# from keras.applications.vgg16 import preprocess_input
# from keras.applications.inception_v3 import preprocess_input
from keras.applications.resnet50 import preprocess_input

from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,Dropout
from datahandler import DataHandlerEyeAgm
from MyIO import IO
import models
import numpy as np
from skimage import io as sio
import tensorflow as tf
import sklearn
import pandas as pd
import os
from keras.callbacks import ModelCheckpoint
cf = tf.ConfigProto()
cf.gpu_options.allow_growth = True

io = IO(__file__, 'train6',inputScript='main4',inputRunName='train6',inputTag=1)
config = io.getRunConfig()
X_train = np.load(io.getInputPath()+'X_train.npy')
X_train = X_train*255
X_train = preprocess_input(X_train)
y_train = np.load(io.getInputPath()+'y_train.npy')
X_test = np.load(io.getInputPath()+'X_test.npy')
y_test = np.load(io.getInputPath()+'y_test.npy')

base_model = ResNet101(weights='imagenet', include_top=False,input_shape=X_train.shape[1:])

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x= Dropout(.2)(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# x= Dropout(.2)(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(4, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)
model.summary()
# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False
for layer in base_model.layers:
    if "BatchNormalization" in layer.__class__.__name__:
        layer.trainable = True

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
model_checkpoint_callback = ModelCheckpoint(
    filepath=io.getOutputPath() + 'model0.{epoch:02d}-{val_loss:.2f}.h5',
    save_weights_only=False,
    monitor='val_loss',
    mode='max',
    save_best_only=False)
my_callbacks = [
    model_checkpoint_callback
    # tf.keras.callbacks.TensorBoard(log_dir='./logs'),
]


# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# train the model on the new data for a few epochs
model.fit(X_train,y_train,epochs=30,validation_split=.2,callbacks=my_callbacks,verbose=2,batch_size=16)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)


for layer in model.layers[0:-11]:
   layer.trainable = False
for layer in model.layers[-11:]:
   layer.trainable = True
for layer in model.layers:
    if "BatchNormalization" in layer.__class__.__name__:
        layer.trainable = True
# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model_checkpoint_callback = ModelCheckpoint(
    filepath=io.getOutputPath() + 'model1.{epoch:02d}-{val_loss:.2f}.h5',
    save_weights_only=False,
    monitor='val_loss',
    mode='max',
    save_best_only=False)
my_callbacks = [
    model_checkpoint_callback
    # tf.keras.callbacks.TensorBoard(log_dir='./logs'),
]
model.fit(X_train,y_train,validation_split=.2,callbacks=my_callbacks,epochs=500,verbose=2,batch_size=16)


