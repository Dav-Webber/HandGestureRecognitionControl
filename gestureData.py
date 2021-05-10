

import math
import cv2
import numpy as np
import pandas as pd
#left 0- right 1
class getData:
    def __init__(self, video, c=None):
        self.protoFile = "hand/pose_deploy.prototxt"
        self.weightsFile = "hand/pose_iter_102000.caffemodel"
        self.nPoints = 22
        self.c = c
        self.POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11],
                      [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
        self.threshold = 0.01
        self.cap = cv2.VideoCapture(video)
        self.numframes = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.framewidth = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.frameheight= self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(self.numframes)
        print(f'frame width: {self.framewidth}')
        print(f'frame height: {self.frameheight}')
        #read frame and turn output to dataframe
        self.allpoints = self.readFrame()
        self.df = self.list2df(self.allpoints)

        self.df=self.scaleAlign(self.df, classnum=self.c)





    def readFrame(self):

        hasFrame, frame = self.cap.read()
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        aspect_ratio = frameWidth / frameHeight

        inHeight = 368
        inWidth = int(((aspect_ratio * inHeight) * 8) // 8)
        net = cv2.dnn.readNetFromCaffe(self.protoFile, self.weightsFile)

        allpoints = []
        k = 1
        while (self.cap.isOpened()):
            print(f'{k} out of {self.numframes} frames!')

            hasFrame, frame = self.cap.read()
            #print(frame.shape)

            if not hasFrame:
                return allpoints

            inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                            (0, 0, 0), swapRB=False, crop=False)
            print(f'inpBlop shape: {inpBlob.shape}')
            net.setInput(inpBlob)
            output = net.forward()

            # Empty list to store the detected keypoints
            points = []

            for i in range(self.nPoints):
                # confidence map of corresponding body's part.
                probMap = output[0, i, :, :]
                probMap = cv2.resize(probMap, (frameWidth, frameHeight))

                # Find global maxima of the probMap.
                minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

                if prob > self.threshold:
                    # Add the point to the list if the probability is greater than the threshold
                    points.append((int(point[0]), int(point[1])))
                else:
                    points.append(None)
            # remove last value as it nothing useful (its probability score)
            del points[-1]
            allpoints.append(points)

            k += 1

        return allpoints


    def list2df(self, listsofpoints):
        inputlist = []
        for row in listsofpoints:
            rowlist = []
            for keypoint in row:
                if keypoint == None:
                    rowlist.append(np.nan)
                else:
                    rowlist.append(keypoint[0])
                    rowlist.append(keypoint[1])
            #print(rowlist)
            if len(rowlist)== 42:
                inputlist.append(rowlist)
            else:
                print('not enough key points detected for this frame')


        df = pd.DataFrame(inputlist,
                          columns=['0x', '0y', '1x', '1y', '2x', '2y', '3x', '3y', '4x', '4y', '5x', '5y',
                                   '6x', '6y', '7x', '7y', '8x', '8y', '9x', '9y', '10x',
                                   '10y', '11x', '11y', '12x', '12y', '13x', '13y', '14x', '14y',
                                   '15x', '15y', '16x', '16y', '17x', '17y', '18x', '18y', '19x', '19y',
                                   '20x', '20y'])
        #print(df)
        return df

    #scale to hand width? align to bottom palm?
    def scaleAlign(self, df, classnum=None):
        #handwidth = 5 to 17
        #bottom of hand = 0
        #coordinates to align all data to
        alignPointX, alignPointY = 800, 400
        refHandwidth = 600
        #workout handwidth input
        nExamples, nColumns = df.shape
        inputDF = df.astype(float)

        dataFrame = inputDF[0:0]
        # body length between neck keypoint and midhip, useful for normalisation

        for i in range(nExamples):
            row = inputDF.iloc[[i]]
            inHandwidth = math.hypot(row.iloc[:, 11].values - row.iloc[:, 35].values,
                                      row.iloc[:, 12].values - row.iloc[:, 36].values)
            #print(inHandwidth)
            differencePCT = inHandwidth / refHandwidth
            print(differencePCT)
            # scale each point in row with the reference
            print(row)
            scaledRow = differencePCT * row

            # align each point
            palmINX, palmINY = scaledRow.iloc[:, 0].values, scaledRow.iloc[:, 1].values

            # maybe change to other way around?
            x_diff = alignPointX - palmINX
            y_diff = alignPointY - palmINY


            scaledRow.iloc[:, ::2] = x_diff + scaledRow.iloc[:, ::2].values
            scaledRow.iloc[:, 1::2] = y_diff + scaledRow.iloc[:, 1::2].values
            alignedRow = scaledRow
            dataFrame = pd.concat([dataFrame, alignedRow])
            if classnum != None:
                dataFrame['class'] = classnum
        dataFrame.to_csv(f'{classnum}scaled30(delete).csv', index=False)
        return dataFrame



getData('videos/right-hand-29.mov', c=1)
getData('videos/left-hand-30.mov', c=0)

df0 = pd.read_csv('0scaled30.csv')
df1 = pd.read_csv('1scaled30.csv')

#Merge, remove unnecessary keypoints
merged= pd.concat([df0, df1], ignore_index=True)
merged = merged.sample(frac=1).reset_index(drop=True)
merged = merged.drop(columns=['0x','0y', '1x', '1y'])

# # #split training and test from dataset 0.8-0.2

df = merged
df['split'] = np.random.randn(df.shape[0], 1)
msk = np.random.rand(len(df)) <= 0.8
train = df[msk]
test = df[~msk]
train = train.drop(columns=['split'])
test = test.drop(columns=['split'])
train.to_csv('train30.csv', index=False)
test.to_csv('test30.csv', index=False)