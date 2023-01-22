import cv2
import glob
import os
import easyocr
import pytesseract
import matplotlib.pyplot as plt
import numpy as np
import imutils

pytesseract.pytesseract.tesseract_cmd=r'D:\APK(s)\tesseract\Installation\tesseract.exe'

def GrayImageTesseract(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return gray

def Preprocessing1Tesseract(grayimage):

    edged = cv2.Canny(grayimage, 30, 200)
    return edged

def ContourTesseract2(image,edged):
    cnts,new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image1=image.copy()
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True) [:10]
    screenCnt = None
    image2 = image.copy()
    cv2.drawContours(image2,cnts,-1,(0,255,0),3)

    return cnts

def countcountoursTesseract(cnts,image):
    i=7
    for c in cnts:
            x,y,w,h = cv2.boundingRect(c) 
            perimeter = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
            if len(approx) == 4: 
                screenCnt = approx
    new_img=image[y:y+h,x:x+w]
    cv2.imwrite('./'+str(i)+'.jpg',new_img)
    i+=1     
    screenCnt

    return screenCnt

def tesseractcontour(image,screenCnt):
    contours = cv2.drawContours(image, [screenCnt], -1, (0, 255, 255), 3)

    return contours

def MaskingTesseract1(gray):
    mask = np.zeros(gray.shape,np.uint8)

    return mask

def MaskTesseract(mask,screenCnt,image):
    new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
    new_image = cv2.bitwise_and(image,image,mask=mask)

    return new_image

def FinalTesseract(mask,gray):
    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    Cropped = gray[topx:bottomx+1, topy:bottomy+1]
    text = pytesseract.image_to_string(Cropped, config='--psm 11')
        # Get the expected text
    expected_text = "This is a sample text"

    # Calculate the accuracy
    accuracy = (sum(1 for x, y in zip(text, expected_text) if x == y) / len(expected_text)) * 100

    # Print the output

    return text, accuracy