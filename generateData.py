import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import csv
import random

#for each gesture
#get set of hand keypoints (left and right separate)
#align and scale (testing input will need to be aligned to the same posture used here)
#get the mean for each part of the hand using all the samples for the same part of the hand
#get std of mean for each part of the hand
#create function that generates random numbers in range of mean +- std of mean

##CSV HEADER (class = 0 for left and 1 for right)
csvHeader =    ['2x', '2y', '3x', '3y', '4x', '4y', '5x', '5y',
                                   '6x', '6y', '7x', '7y', '8x', '8y', '9x', '9y', '10x',
                                   '10y', '11x', '11y', '12x', '12y', '13x', '13y', '14x', '14y',
                                   '15x', '15y', '16x', '16y', '17x', '17y', '18x', '18y', '19x', '19y',
                                   '20x', '20y']


#0 left gesture 1 right gesture
DF0 = pd.read_csv('0scaled30.csv')
DF0 = DF0.drop(columns=['class'])
DF0 = DF0.replace(0, np.nan)
DF1 = pd.read_csv('1scaled30.csv')
DF1 = DF1.drop(columns=['class'])
DF1 = DF1.replace(0, np.nan)

DF0= DF0.drop(columns=['0x','0y', '1x', '1y'])
DF1= DF1.drop(columns=['0x','0y', '1x', '1y'])


#num2gen = to specify how many rows of data to generate
#binary = 0 if left and 1 if rightt
def augmentData(inputDF, num2gen,binary):
    #the number of different postures in input and columns
    nExamples, nColumns = inputDF.shape
    meanDF = inputDF.mean(axis=0)
    stdDF = inputDF.std(axis=0)
    lowR = meanDF - stdDF
    highR = meanDF + stdDF
    # print("LOW ranges: ")
    # print(lowR)
    # print("High ranges: ")
    # print(highR)

    for i in range(num2gen):
        gData = random.uniform(lowR,highR)
        #1 = right, 0 = left
        classNum = pd.Series([binary])
        gData = gData.append(classNum)
        #prints row to csv file
        wr.writerow(gData)
        print(i)

with open("synthetic-0.csv", "a") as fp:
    wr = csv.writer(fp, dialect='excel')
    wr.writerow(csvHeader)
    #synthesize 5000 rows of data for class 0
    augmentData(DF0, 5000,0)