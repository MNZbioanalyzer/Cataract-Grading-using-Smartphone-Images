# eye extraction with regenerate total face
import dlib
from skimage import io as sio
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
from MyIO import IO
from fnmatch import fnmatch
import numpy as np

io = IO(__file__, 'eye_detection')
config = io.getRunConfig()
file_dirs = ['pre','post']
pattern = "*.jpg"
image_files = []
targets = pd.read_excel('data/targets.xlsx', header=1)
targets.set_index('Folder name', inplace=True)
targets.rename(columns={'Right Eye': 'R', 'Left Eye': 'L'}, inplace=True)
for dir in file_dirs:
    image_files = []
    try:
        os.makedirs(os.path.dirname(io.getPlotsOutputPath()+dir+'/'))
    except:
        pass
    try:
        os.makedirs(os.path.dirname(io.getPlotsOutputPath()+'ss/'))
    except:
        pass

    for path, subdirs, files in os.walk('data/'+dir):
        for name in files:
            if fnmatch(name, pattern):
                image_files.append(os.path.join(path, name))
    counter = 0
    df = pd.DataFrame(columns=['input_file','eye_detected','accurate_eye','class'])
    df['accurate_eye'] = 0
    df['eye_detected'] = 0
    df['input_file'] = image_files
    df.set_index('input_file',inplace=True)
    for f in image_files:
        subject = f.split('\\')[-1]
        isRight = True if subject.split('_')[2]=='R' else False
        try:
            target = targets.loc[int(subject.split('_')[1]),subject.split('_')[2]]
            df.loc[f, 'class'] = target
        except:
            pass
        try:
            counter += 1
            img = sio.imread(f)
            if img.shape[0] > img.shape[1]:
                img = cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)
            img = cv2.resize(img, (img.shape[0]//2, img.shape[1]//2))
            if isRight:
                img = cv2.hconcat([img,cv2.flip(img, 1) ])

            else:
                img = cv2.hconcat([cv2.flip(img, 1), img ])

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rec = dlib.rectangle(0,0,gray.shape[1],gray.shape[0])

            detector = dlib.get_frontal_face_detector()
            detect = detector(gray, 1)
            predictor = dlib.shape_predictor(
                "data/shape_predictor_68_face_landmarks.dat" )
            shape = predictor(gray, rec)

            try:
                x1 = shape.part(44).x
                x2 = shape.part(47).x
                y1 = shape.part(43).y
                y2 = shape.part(46).y
                mid_x = (x1+x2)//2
                mid_y = (y2 + y1)//2

                recXSize = 600
                recYSize = 800
                plt.figure()
                sio.imshow(img)
                bias = img.shape[0]
                for i in range(43, 48):
                    plt.plot(img.shape[1]-shape.part(i).x,shape.part(i).y, '*')
                plt.savefig(io.getPlotsOutputPath()+'ss/'+subject+'.jpg')
                plt.close()
                eye = img[max(bias-(mid_x - recXSize),0):min(bias-(mid_x + recXSize),img.shape[0]),max(mid_y - recYSize,0):min(mid_y + recYSize,img.shape[1])]
                eye =  cv2.resize(eye, (eye.shape[1]//2, eye.shape[0]//2))

                ht, wd, cc = eye.shape
                # create new image of desired size and color (blue) for padding
                ww = 800
                hh = 600
                color = (0, 0, 0)
                result = np.full((hh, ww, cc), color, dtype=np.uint8)

                # compute center offset
                xx = (ww - wd) // 2
                yy = (hh - ht) // 2

                # copy img image into center of result image
                result[yy:yy + ht, xx:xx + wd] = eye
                image_to_write = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

                cv2.imwrite(io.getPlotsOutputPath()+dir+'/' + subject,
                            image_to_write)
                df.loc[f, 'eye_detected'] = 1
                print(counter.__str__() + ': OK!!')

            except Exception as err:
                print('1')
                print(err.__str__())
                df.loc[f, 'eye_detected'] = 0
                print(counter.__str__() + ': Failed!!')
        except Exception as err:
            print('2')
            print(err.__str__())
            df.loc[f, 'eye_detected'] = 0
            print(counter.__str__() + ': Failed!!')

    df.to_csv(io.getOutputPath()+'res_'+dir+'.csv')
print('end')
