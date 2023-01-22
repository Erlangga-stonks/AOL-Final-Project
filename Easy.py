import cv2
import os
import easyocr
import pytesseract
import matplotlib.pyplot as plt
import numpy as np
import imutils

def GrayImage(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return gray

def Preprocessing1(grayimage):
    bfilter = cv2.bilateralFilter(grayimage, 11, 17, 17) 
    edged = cv2.Canny(bfilter, 30, 200)
    return edged

def threshold(img):
    res,thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    return thresh

def Contours(thresh):
    keypoints = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]

    return contours

def locationEasyOCR(contours,img):
    location = None
    for contour in contours:
            # peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 10 , True)
        if len(approx) == 4:
            location = approx
            break
    if location is None:
        detected = 0
        print ("No contour detected")
    else:
        detected = 1
    if detected == 1:
        cv2.drawContours(img, [location], -1, (0, 255, 0), 6)

    return location

def masking0(gray):
    mask = np.zeros(gray.shape, np.uint8)

    return mask

def masking(mask,img,location):                
    new_image = cv2.drawContours(mask, [location], 0,255, 255)
    new_image = cv2.bitwise_and(img, img, mask=mask)

    return new_image

def resultmasking(gray,mask):
    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    Cropped = gray[topx:bottomx+1, topy:bottomy+1]
    
    return Cropped

def ReadEasyOCR(Cropped):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(Cropped)
    
    return result

def FinalResult(result):
    text = result[0][-2]
    text2 = result[0][-1]

    return text, text2

def FinalEasyOCR(img,result,contours):
    text = result[0][-2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    location = None
    for contour in contours:
            # peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 10 , True)
        if len(approx) == 4:
            location = approx
            break
    if location is None:
        detected = 0
        print ("No contour detected")
    else:
        detected = 1
    if detected == 1:
        cv2.drawContours(img, [location], -1, (0, 255, 0), 6)
    res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
    res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)

    return res
