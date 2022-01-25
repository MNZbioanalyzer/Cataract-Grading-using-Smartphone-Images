import pandas  as pd
import numpy as np
import os
from skimage import io
from fnmatch import fnmatch
import random
from sklearn.model_selection import KFold
from skimage.transform import rescale, resize, downscale_local_mean,rotate
from sklearn.model_selection import train_test_split
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array

class DataHandler():
    def __init__(self, config):
        self.config = config
        self.idx = pd.IndexSlice
        self.pre_files_dir = self.config.getStringConfig('PRE_IMAGES_DIR')
        self.post_files_dir = self.config.getStringConfig('POST_IMAGES_DIR')
        # self.splitData(self.total_data,self.y)
        self.init_data()

    def init_data(self):
        if self.config.getBooleanConfig('LOAD_DATA_FROM_NPY'):
            self.X = np.load('data/X.npy')
            self.y = np.load('data/y.npy')
        else:
            self.pre_data = self.loadDataFromFiles(self.pre_files_dir)
            self.post_data = self.loadDataFromFiles(self.post_files_dir)
            self.X = np.concatenate([self.pre_data, self.post_data])
            self.y = np.concatenate([np.ones(self.pre_data.shape[0]), np.zeros([self.post_data.shape[0]])])
            if self.config.getBooleanConfig('SAVE_NPY_DATA'):
                np.save("data/X.npy", self.X)
                np.save("data/y.npy", self.y)


        return self.X,self.y

    def loadDataFromFiles(self,file_dir):
        pattern = "*.jpg"
        image_files = []
        for path, subdirs, files in os.walk(file_dir):
            for name in files:
                if fnmatch(name, pattern):
                    image_files.append(os.path.join(path, name))

        counter = 0
        for f in image_files:
            if counter ==0:
                img = io.imread(f)
                # print(img.shape)
                img = rgb2gray(img)
                img = rescale(img, self.config.getFloatConfig('SCALE_FACTOR'), anti_aliasing=False)
                result = np.empty([image_files.__len__(), img.shape[0], img.shape[1]])
                result[counter, :, :] = img
            else:
                img = io.imread(f)
                # print(img.shape)
                img = rgb2gray(img)
                img = rescale(img, self.config.getFloatConfig('SCALE_FACTOR'), anti_aliasing=False)
                try:
                    result[counter,:,:] = img
                except Exception as err:
                    # img = rotate(img,90)
                    result[counter,:,:] = img.T

                    # print(f + '     error ')
            counter+=1
            print(counter.__str__())
        return result

    def splitData(self,X,y):
        typ = self.config.getStringConfig('SPLIT_TYPE')
        if  typ== 'KFOLD':
            self.kf = KFold(n_splits=10,shuffle=True)
            self.kf.get_n_splits(self.X)
            pass
        elif typ == 'TRAIN_TEST':
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        return X_train,X_test,y_train,y_test

    def getTrainData(self):
        pass

    def getTestData(self):
        pass


class DataHandlerEye():
    def __init__(self, io):
        self.io = io
        self.config = io.getRunConfig()
        self.idx = pd.IndexSlice
        self.init_data()

    def init_data(self):
        if self.config.getBooleanConfig('LOAD_DATA_FROM_NPY'):
            self.X = np.load('data/eyes2_X.npy')
            self.y = np.load('data/eyes2_y.npy')
        else:
            self.loadDataFromFiles()



        return self.X,self.y

    def loadDataFromFiles(self):
        pattern = "*.jpg"
        image_files = []
        targets_pre = pd.read_csv(self.io.getInputPath()+'res_pre.csv',header=0)
        targets_post = pd.read_csv(self.io.getInputPath()+'res_post.csv',header=0)
        targets = pd.concat([targets_pre,targets_post],axis=0)
        targets['subject'] = targets['input_file'].apply(lambda x: x.split('\\')[-1].split('.')[0])
        for path, subdirs, files in os.walk(self.io.getInputPath()+'/figures/'):
            for name in files:
                if fnmatch(name, pattern):
                    image_files.append(os.path.join(path, name))

        counter = 0
        bad_images = os.listdir(self.io.getInputPath()+'bad/')
        bad_images1 = os.listdir(self.io.getInputPath()+'bad1/')
        for f in image_files:
            subject = f.split('\\')[-1].split('.')[0]
            if (not subject+'.jpg' in bad_images) and (not subject+'.jpg' in bad_images1):
                if counter ==0:
                    img = io.imread(f)
                    img = cv2.resize(img,(img.shape[0]//2,img.shape[1]//2))
                    result = np.empty([image_files.__len__(), img.shape[0], img.shape[1],3],dtype='uint8')
                    y = pd.DataFrame(index = range(image_files.__len__()),columns=['target'])
                    y.loc[counter,'target'] = targets.loc[targets['subject']==subject,'class'].values[0]
                    result[counter, :, :,:] = img_to_array(img)
                else:
                    img = io.imread(f)
                    img = cv2.resize(img,(img.shape[0]//2,img.shape[1]//2))
                    result[counter, :, :,:] = img_to_array(img)
                    y.loc[counter,'target'] = targets.loc[targets['subject']==subject,'class'].values[0]
                counter+=1
                print(counter.__str__())

        self.X = result
        self.y = y
        return self.X,self.y


    def clearData(self):
        valid_data = self.y.loc[self.y['target'].isin(['NC','EC','PMC','MC'])].index
        self.y = self.y.loc[valid_data]
        self.X = self.X[valid_data,:,:,:]

    def splitData(self,X,y,test_size=.25):
        typ = self.config.getStringConfig('SPLIT_TYPE')
        if  typ== 'KFOLD':
            self.kf = KFold(n_splits=10,shuffle=True)
            self.kf.get_n_splits(self.X)
            pass
        elif typ == 'TRAIN_TEST':
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        return X_train,X_test,y_train,y_test

    def getTrainData(self):
        pass

    def getTestData(self):
        pass

    def createDataGenerator(self,X,y):
        datagen = ImageDataGenerator(width_shift_range=[-50,50],height_shift_range=[-30,30],horizontal_flip=True,vertical_flip=True,rotation_range=90,brightness_range=[0.7,1.3],validation_split=0.2)
        it = datagen.flow(X,y)
        return it

    def createPureDataGenerator(self,X,y):
        datagen = ImageDataGenerator()
        it = datagen.flow(X,y)
        return it




class DataHandlerEyeAgm():
    def __init__(self, io):
        self.io = io
        self.config = io.getRunConfig()
        self.idx = pd.IndexSlice
        self.init_data()

    def init_data(self):
        if self.config.getBooleanConfig('LOAD_DATA_FROM_NPY'):
            self.X = np.load('data/eyes2_X.npy')
            self.y = np.load('data/eyes2_y.npy')
        else:
            self.loadDataFromFiles()



        return self.X,self.y

    def loadDataFromFiles(self):
        pattern = "*.jpg"
        image_files = []
        targets_pre = pd.read_csv(self.io.getInputPath()+'res_pre.csv',header=0)
        targets_post = pd.read_csv(self.io.getInputPath()+'res_post.csv',header=0)
        targets = pd.concat([targets_pre,targets_post],axis=0)
        targets['subject'] = targets['input_file'].apply(lambda x: x.split('\\')[-1].split('.')[0])
        for path, subdirs, files in os.walk(self.io.getInputPath()+'/figures/'):
            for name in files:
                if fnmatch(name, pattern):
                    image_files.append(os.path.join(path, name))

        counter = 0
        bad_images = os.listdir(self.io.getInputPath()+'bad/')
        bad_images1 = os.listdir(self.io.getInputPath()+'bad1/')
        for f in image_files:
            subject = f.split('\\')[-1].split('.')[0].split('_',1)[1]
            if (not subject+'.jpg' in bad_images) and (not subject+'.jpg' in bad_images1):
                if counter ==0:
                    img = io.imread(f)
                    img = cv2.resize(img,(img.shape[0]//2,img.shape[1]//2))
                    result = np.empty([image_files.__len__(), img.shape[0], img.shape[1],3],dtype='uint8')
                    y = pd.DataFrame(index = range(image_files.__len__()),columns=['target'])
                    ids = pd.DataFrame(index = range(image_files.__len__()),columns=['subject'])
                    y.loc[counter,'target'] = targets.loc[targets['subject']==subject,'class'].values[0]
                    ids.loc[counter,'subject'] = subject
                    result[counter, :, :,:] = img_to_array(img)
                else:
                    img = io.imread(f)
                    img = cv2.resize(img,(img.shape[0]//2,img.shape[1]//2))
                    result[counter, :, :,:] = img_to_array(img)
                    y.loc[counter,'target'] = targets.loc[targets['subject']==subject,'class'].values[0]
                    ids.loc[counter,'subject'] = subject

                counter+=1
                print(counter.__str__())

        self.X = result
        self.y = y
        self.ids = ids
        return self.X,self.y


    def clearData(self):
        valid_data = self.y.loc[self.y['target'].isin(['NC','EC','PMC','MC'])].index
        self.y = self.y.loc[valid_data]
        self.X = self.X[valid_data,:,:,:]
        self.ids = self.ids.loc[self.y.index]
        self.ids['target'] = np.argmax(self.y.values,axis=1)
        self.ids['pure_subject'] = self.ids['subject'].apply(lambda x:"_".join(x.split('_',1)[1].split("_", 2)[:2]))

    def splitData(self,X,y,test_size=.25):
        typ = self.config.getStringConfig('SPLIT_TYPE')
        if  typ== 'KFOLD':
            self.kf = KFold(n_splits=10,shuffle=True)
            self.kf.get_n_splits(self.X)
            pass
        elif typ == 'TRAIN_TEST':
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        elif typ == 'TRAIN_TEST_SUBJECTIVE':
            subjects = list(self.ids['pure_subject'].unique())
            test_subjects = random.sample(subjects,k=int(subjects.__len__()*test_size))
            if self.ids.loc[self.ids['pure_subject'].isin(test_subjects),'target'].unique().__len__()<self.ids['target'].unique().__len__():
                return self.splitData(X,y,test_size)
            else:
                train_subjects = list(set(subjects).difference(set(test_subjects)))
                y_train = y.loc[self.ids['pure_subject'].isin(train_subjects)]
                y_test =  y.loc[self.ids['pure_subject'].isin(test_subjects)]
                self.ids['index'] = self.ids.reset_index().index
                X_train = X[self.ids.loc[self.ids['pure_subject'].isin(train_subjects),'index'].values,:,:,:]
                X_test = X[self.ids.loc[self.ids['pure_subject'].isin(test_subjects),'index'].values,:,:,:]

        return X_train,X_test,y_train,y_test

    def getTrainData(self):
        pass

    def getTestData(self):
        pass

    def createDataGenerator(self,X,y):
        datagen = ImageDataGenerator()
        it = datagen.flow(X,y)
        return it

    def createPureDataGenerator(self,X,y):
        datagen = ImageDataGenerator()
        it = datagen.flow(X,y)
        return it