
import tkinter
from tkinter.ttk import *
import PIL.Image, PIL.ImageTk
import cv2
import joblib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn import preprocessing
import math
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.naive_bayes import GaussianNB



class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.resizable(True, True)
        self.window.title(window_title)
        self.video_source = video_source

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)
        #start game
        self.game = Game()

        self.gesture = Gesture()

        # Create a canvas that can fit the above video source size
        self.camcanvas = tkinter.Canvas(window, width = 500, height = 500)
        self.camcanvas.pack(side = 'left')
        # Create a canvas that can fit the above game
        self.gamecanvas = tkinter.Canvas(window, width=200, height=200,highlightbackground='#3E4149')
        self.gamecanvas.pack(side = 'right')
        #Create a canvas that can fit a hand pose detected image
        self.imgcanvas = tkinter.Canvas(window, width = 500, height = 500)
        self.imgcanvas.pack(side = 'top')
        #self.game.movement('right')



        # Button that lets the user take a snapshot
        self.btn_snapshot=tkinter.Button(window, text="Snapshot", width=25, command=self.snapshot,highlightbackground='#3E4149')
        self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)

        # quit button
        self.btn_quit = tkinter.Button(window, text="QUIT", highlightbackground='#3E4149',command=self.window.destroy)
        self.btn_quit.pack(side="bottom")

        # text info
        self.text_widget = tkinter.Text(window, height=2, width= 50)
        self.text_widget.pack()
        self.text_widget.insert(tkinter.END, 'take a snapshot of your right hand pointing right, or left hand pointing left to move the box')

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        #update cam
        self.updateCam()
        #updategame
        #print(f'gesture value: {self.gesture.value}')
        self.block = self.gamecanvas.create_rectangle(self.game.x1, self.game.y1, self.game.x2, self.game.y2,
                                                 fill='blue', tags='block')
        self.updateGame(self.gesture.value)

        self.window.mainloop()



    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        self.getGesture(frame)
        self.updatePose(frame)
        ##saves to image file
        # if ret:
        #     cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def updateCam(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        ##reseize frame to fit in canvas NEW!
        frame = cv2.resize(frame, (654,368),interpolation =cv2.INTER_AREA)
        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.camcanvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)

        self.window.after(self.delay, self.updateCam)

    def updatePose(self, frame):
        # Draw Skeleton
        ##resize frame to fit in canvas NEW!
        frame = cv2.resize(frame, (654, 368), interpolation=cv2.INTER_AREA)
        for pair in self.gesture.POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]

            if self.gesture.points[partA] and self.gesture.points[partB]:
                cv2.line(frame, self.gesture.points[partA], self.gesture.points[partB], (0, 255, 255), 2, lineType=cv2.LINE_AA)
                cv2.circle(frame, self.gesture.points[partA], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(frame, self.gesture.points[partB], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)


        self.photopose = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
        self.imgcanvas.create_image(0, 0, image = self.photopose, anchor = tkinter.NW)

    def updateGame(self, value):
        if value == 0:
            self.game.movement('left')
        elif value == 1:
            self.game.movement('right')
        elif value =='s':
            self.game.movement('down')
        elif value == 'w':
            self.game.movement('up')
        self.gesture.value = -1
        self.gamecanvas.delete(self.block)
        self.block = self.gamecanvas.create_rectangle(self.game.x1, self.game.y1, self.game.x2, self.game.y2,
                                                      fill='blue', tags='block')

    def getGesture(self, frame):
        #resize to the same resolution as was used for the training model
        #frame = cv2.resize(frame,(720,1280),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
        #frame = cv2.resize(frame, (720, 1280), fx=0, fy=0, interpolation=cv2.INTER_AREA)

        frameCopy = np.copy(frame)
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        aspect_ratio = frameWidth / frameHeight

        inHeight = 368
        # inWidth = int(((aspect_ratio * inHeight) * 8) // 8)
        inWidth = 654

        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                        (0, 0, 0), swapRB=False, crop=False)
        print(f'inpBlop shape: {inpBlob.shape}')
        ##TODO: is opencv compatible with pytorch, setInput works?
        self.gesture.net.setInput(inpBlob)
        output = self.gesture.net.forward()

        ##next part
        # Empty list to store the detected keypoints

        self.gesture.points = []

        for i in range(self.gesture.nPoints):
            # confidence map of corresponding body's part.
            probMap = output[0, i, :, :]
            # was 138 not 139
            #probMap = cv2.resize(probMap, (frameWidth, frameHeight))
            probMap = cv2.resize(probMap, (inWidth, inHeight))
            #probMap = cv2.resize(probMap, (300, 300))


            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            if prob > self.gesture.threshold:
                cv2.circle(frameCopy, (int(point[0]), int(point[1])), 6, (0, 255, 255), thickness=-1,
                           lineType=cv2.FILLED)
                cv2.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, .8,
                            (0, 0, 255), 2, lineType=cv2.LINE_AA)

                # Add the point to the list if the probability is greater than the threshold
                self.gesture.points.append((int(point[0]), int(point[1])))
            else:
                self.gesture.points.append(None)
        #TODO:fix output, strange output, generate new test data to analyse outputs
        # and accuracy with the model

        dfScaled = self.scaleAlign(self.list2df(self.gesture.points))
        tensor = self.scaled2tensor(dfScaled)
        print(tensor)
        #load gesture classifier model
        device = torch.device("cpu")
        net = Net(38).to(device)

        PYTORCH_NN_PATH = 'models/gestureModel_net.pth'
        TREE_PATH = 'models/tree_model.sav'
        GNB_PATH = 'models/GNB_model.sav'
        SKL_NN_PATH = 'models/SKL_NN_model.sav'
        SKL_SCALER_PATH = 'models/SKL_scaler.bin'
        #load NN model
        net.load_state_dict(torch.load(PYTORCH_NN_PATH))
        for parameter in net.parameters():
            parameter.requires_grad = False
        #print(net.state_dict())
        net.eval()
        torch.manual_seed(0)
        #Load tree model
        loaded_tree_model = joblib.load(TREE_PATH)
        tree_prediction = loaded_tree_model.predict(dfScaled.values)
        print(f'Gesture value Tree: {tree_prediction}')
        #Load naive bayes model
        loaded_gnb_model = joblib.load(GNB_PATH)
        gnb_prediction = loaded_gnb_model.predict(dfScaled.values)
        print(f'Gesture value Naive Bayes: {gnb_prediction}')
        #load skl NN model and scaler
        loaded_skl_nn_model = joblib.load(SKL_NN_PATH)
        loaded_skl_scaler = joblib.load(SKL_SCALER_PATH)
        ##scale data to pass to model
        scaler_values = loaded_skl_scaler.transform(dfScaled.values)
        #predict
        skl_nn_prediction = loaded_skl_nn_model.predict(scaler_values)
        print(f'Gesture value SKL NN: {skl_nn_prediction}')

        #simple hard voting ensemble
        classifier_output_array = np.array([tree_prediction[0], gnb_prediction[0], skl_nn_prediction[0]])

        counts = np.bincount(classifier_output_array)
        ensemble_output = np.argmax(counts)
        print(ensemble_output)

        ##TODO results not good with pytorch nn, not sure why

        output = net(tensor)
        print(round(output.item(), 10))
        if output.item() > 0.5:

            result = 1
        else:

            result = 0
        #self.gesture.value = result
        print(f'Gesture value NN: {result}')
        self.gesture.value = ensemble_output
        self.updateGame(self.gesture.value)


    #minmax
    def scaled2tensor(self, df):
        # normalize
        df = df
        # df = pd.read_csv('handdataset4.csv')
        # df = df.iloc[263]  #should be 0
        # df = df.drop(labels=['class'])
        # df = (df.to_frame()).T
        # print(df)

        xval = df.values
        #print(xval)
        #min_max_scaler = preprocessing.MinMaxScaler()
        #load up scaler
        scaler_filename = "models/min_max_scaler.save"
        min_max_scaler = joblib.load(scaler_filename)
        #minmaxscale from saved scaler used on training model
        xval_scaled = min_max_scaler.transform(xval)
        # convert to torch tensors
        xval_scaled = pd.DataFrame(xval_scaled)
        tensor = torch.tensor(xval_scaled.to_numpy()).float()
        return Variable(tensor).float()#was just tensor

    #preprocess list and make snapshot pose to dataframe
    def list2df(self, listofpoints):
        #remove last value as it nothing useful (its probability score)
        del listofpoints[-1]

        inputlist = []
        for keypoint in listofpoints:
            if keypoint==None:
                #remember to replace all -1 with nan after
                inputlist.append(np.nan)
            else:
                inputlist.append(keypoint[0])
                inputlist.append(keypoint[1])
        inputlist = [inputlist]

        df = pd.DataFrame(inputlist,columns=['0x', '0y', '1x', '1y', '2x', '2y', '3x', '3y', '4x', '4y', '5x', '5y',
                                   '6x', '6y', '7x', '7y', '8x', '8y', '9x', '9y', '10x',
                                   '10y', '11x', '11y', '12x', '12y', '13x', '13y', '14x', '14y',
                                   '15x', '15y', '16x', '16y', '17x', '17y', '18x', '18y', '19x', '19y',
                                   '20x', '20y'])

        #print(df)
        return df

    # scale to hand width - align to bottom palm
    def scaleAlign(self, df):
        # handwidth = 5 to 17
        # bottom of hand = 0
        # coordinates to align all data to
        alignPointX, alignPointY = 800, 400
        refHandwidth = 600
        # workout handwidth input
        nExamples, nColumns = df.shape
        inputDF = df.astype(float)
        dataFrame = inputDF[0:0]

        for i in range(nExamples):
            row = inputDF.iloc[[i]]
            inHandwidth = math.hypot(row.iloc[:, 11].values - row.iloc[:, 35].values,
                                     row.iloc[:, 12].values - row.iloc[:, 36].values)
            differencePCT = inHandwidth / refHandwidth
            # scale each point in row with the reference
            scaledRow = differencePCT * row

            # align each point
            palmINX, palmINY = scaledRow.iloc[:, 0].values, scaledRow.iloc[:, 1].values
            x_diff = alignPointX - palmINX
            y_diff = alignPointY - palmINY

            scaledRow.iloc[:, ::2] = x_diff + scaledRow.iloc[:, ::2].values
            scaledRow.iloc[:, 1::2] = y_diff + scaledRow.iloc[:, 1::2].values
            alignedRow = scaledRow
            dataFrame = pd.concat([dataFrame, alignedRow])
            dataFrame = dataFrame.drop(columns=['0x','0y', '1x', '1y'])
        return dataFrame



class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(self.width, self.height)

        # self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
        # #self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 654)
        # #self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 368)





    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                #frame = cv2.flip(frame, 1)

                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


class Game:
    def __init__(self):
        # box starting positon
        self.x1, self.y1, self.x2, self.y2 = 25, 50, 50, 75
        self.boxwidth = 25

    def movement(self, position):
        if position == 'right':
            #self.gamecanvas.move(self.block, x, y)
            self.x1 = self.x1 + self.boxwidth
            self.x2 = self.x2 + self.boxwidth
        elif position =='left':
            #self.gamecanvas.move(self.block, -self.boxwidth, 0)
            self.x1 = self.x1 - self.boxwidth
            self.x2 = self.x2 - self.boxwidth
        elif position == 'down':
            #self.gamecanvas.move(self.block, 0, self.boxwidth)
            self.y1 = self.y1 + self.boxwidth
            self.y2 = self.y2 + self.boxwidth
        elif position == 'up':
            #self.gamecanvas.move(self.block, 0, -self.boxwidth)
            self.y1 = self.y1 - self.boxwidth
            self.y2 = self.y2 - self.boxwidth

class Gesture:
    def __init__(self):
        self.value = -1
        self.protoFile = "hand/pose_deploy.prototxt"
        self.weightsFile = "hand/pose_iter_102000.caffemodel"
        self.nPoints = 22
        self.POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11],
                      [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
        self.threshold = 0.01
        self.net = cv2.dnn.readNetFromCaffe(self.protoFile, self.weightsFile)
        self.points = []

#gesture classification model
class Net(nn.Module):

    def __init__(self, n_features):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, 38)
        self.fc2 = nn.Linear(38, 16)
        self.fc3 = nn.Linear(16, 1)
        # self.fc1 = nn.Linear(n_features, 12)
        # self.fc2 = nn.Linear(12, 4)
        self.softmax = nn.Softmax(dim=1)
        #self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        #x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc3(x))
        #x = self.fc3(x)
        #x = self.softmax(x)
        #x = F.relu(self.fc3(x))
        print(f'forward net output: {x}')
        return x






# create window
App(tkinter.Tk(), "Gesture recognition to move blue box left or right")

