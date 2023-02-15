from turtle import width
import cv2
from cv2 import FONT_HERSHEY_COMPLEX
import numpy as np
import utilities
from PIL import Image
import glob
import ast

widthImg = 600
heightImg = 2100
questions = 30
choices = 4
ansKey = [int(i) for i in '111111111111111111111111111111']

class Style:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


def operations(img):


    imgFinal = img.copy()


    ############### Image preprocessing ###################


    img = cv2.resize(img, (widthImg, heightImg))
    imgContours = img.copy()
    imgBiggestContours = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv2.Canny(imgBlur, 10,20)
    imgArray = ([img, imgGray, imgBlur, imgCanny])

    stackedImage = utilities.stackImages(imgArray,0.5)
    cv2.imshow("Result",stackedImage)

    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(imgContours, contours, -1,(0,255,0),10)

    #Rectangle

    rectCon = utilities.rectContour(contours)
    biggestContour = utilities.getCornerPoints(rectCon[0])

    if biggestContour.size != 0:
        cv2.drawContours(imgBiggestContours, biggestContour,-1,(0,255,0), 10)
        biggestContour = utilities.reorder(biggestContour)
        
        point1 = np.float32(biggestContour)
        point2 = np.float32([[0,0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
        matrix = cv2.getPerspectiveTransform(point1, point2)
        imgWarpColoured = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
        
        
        #Find out marked answers
        imgWarpGray = cv2.cvtColor(imgWarpColoured, cv2.COLOR_BGR2GRAY)
        imgThresh = cv2.threshold(imgWarpGray, 180,255, cv2.THRESH_BINARY_INV)[1]
        
        boxes = utilities.splitOptions(imgThresh)
        
        ## Differentiating pixels
        pixelVal = np.zeros((questions,choices))
        colCount = 0
        rowCount = 0
        
        
        for image in boxes:
            totalPixels = cv2.countNonZero(image)
            pixelVal[rowCount][colCount] = totalPixels
            colCount += 1
            if (colCount == choices):
                rowCount += 1
                colCount = 0
        
        # DETECTING MARKED AND UNMARKED ANSWERS
        
        myIndex = []
        for x in range(0,questions):
            arr = pixelVal[x]
            arr1a = arr.tolist()
            arr1 = [int(d) for d in arr1a]
            markedOption = min(arr1) + 1080
            for j in range(len(arr1)):
                count = 0
                if arr[j] > markedOption:
                    myIndex.append(j)
                else:
                    myIndex.append('N')
                    
        markedMap = [myIndex[i:i+4] for i in range(0, len(myIndex), 4)]
        removeDupes = []
        marked = []
        for i in range(len(markedMap)):
            removeDupes.append(list(set(markedMap[i])))
        #print(removeDupes)
        lenRem = []
        for i in range(len(removeDupes)):
            lenRem.append(len(removeDupes[i]))
                
        #print(lenRem)
        
        for i in range(len(lenRem)):
            if lenRem[i] > 2:
                marked.append('D')
            if lenRem[i] == 2:
                marked.append(removeDupes[i][0])
            elif lenRem[i] == 1:
                marked.append('N')
        
        print(marked)
        
        
        
        #print(len(myIndex))
        
        #grading
        
        
        grade = []
        score = 0
        correctCount = 0
        wrongCount = 0
        unmarked = 0
        disqualified = 0
        
        for x in range(0,questions):
            if marked[x] == 'N':  # No options marked
                score = score + 0
                unmarked += 1
                
            elif marked[x] == 'D':  # More than one options marked. 
                score = score - 0.5
                disqualified += 1
                
            elif ansKey[x] == marked[x]:  # Correct answer
                score = score + 1
                correctCount += 1
                grade.append(1)
                
            elif marked[x] != 'N' and marked[x] != 'D' and ansKey[x] != marked[x]:  # Wrong answer
                score = score - 0.25
                wrongCount += 1
                grade.append(0)

        print('\n')
        print("Correct ans = ", correctCount)
        print("Wrong ans = ", wrongCount)
        print("Unmarked = ", unmarked)
        print("Disqualified = ", disqualified)
        print("Total score = ", score)
        

    cv2.imshow("Stacked", imgThresh)
    cv2.imshow("Grade",imgFinal)
    cv2.waitKey(0)

img = cv2.imread("y2.JPG")
operations(img)