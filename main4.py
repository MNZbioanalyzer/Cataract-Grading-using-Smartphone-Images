# handly agumented data + pre-trainded VGG16
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,Dropout
from datahandler import DataHandlerEyeAgm
from MyIO import IO
import numpy as np
import pandas as pd
import os
from keras.callbacks import ModelCheckpoint

io = IO(__file__, 'train4',inputScript='eye_detection_agumentation',inputRunName='eye_detection',inputTag=1)
config = io.getRunConfig()
dataHandler = DataHandlerEyeAgm(io)
dataHandler.clearData()
dataHandler.y = pd.get_dummies(dataHandler.y['target'])
dataHandler.X = dataHandler.X/255.0
X_train, X_test, y_train, y_test = dataHandler.splitData(dataHandler.X,dataHandler.y,test_size=.2)
np.save(io.getOutputPath()+'X_train.npy',X_train)
np.save(io.getOutputPath()+'X_test.npy',X_test)
np.save(io.getOutputPath()+'y_train.npy',y_train)
np.save(io.getOutputPath()+'y_test.npy',y_test)

# create the base pre-trained model
base_model = VGG16(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
x= Dropout(.1)(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(4, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

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
model.fit(X_train,y_train,epochs=50,validation_split=.2,callbacks=my_callbacks,verbose=2)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:-5]:
   layer.trainable = False
for layer in model.layers[-5:]:
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
model.fit(X_train,y_train,validation_split=.2,callbacks=my_callbacks,epochs=300,verbose=2)


