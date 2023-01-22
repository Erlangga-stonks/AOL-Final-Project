import streamlit as st
import numpy as np
import cv2
from  PIL import Image, ImageEnhance
from io import BytesIO

from Easy import *
from Tersseract import *

st.set_page_config(
    page_title="Plate Recognition App",
    page_icon="😘",
    )

st.title("Plate Recognition App")
st.text("")
"""
Please Input The Image
"""
uploaded_file = st.file_uploader("", type=['jpg','png','jpeg'])

#def Preprocessing(img):
#    image2 = np.array(img.convert('RGB'))
#    image3 = cv2.cvtColor(image2,1)
#    gray = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)
#    bfilter = cv2.bilateralFilter(gray, 11, 17, 17) 
#    edged = cv2.Canny(bfilter, 30, 200)
#    return edged

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2, col3 = st.columns( [0.5, 0.5,0.5])
    with col1:
        st.markdown('<p style="text-align: center;color: white;">EasyOCR</p>',unsafe_allow_html=True)
        image2 = np.array(image.convert('RGB'))
        image3 = cv2.cvtColor(image2,1)
        colorgray = GrayImage(image3)
        prep = Preprocessing1(colorgray)
        thres = threshold(prep)
        SearchCountour = Contours(thres)
        Con = locationEasyOCR(SearchCountour,image3)
        maskduh = masking0(prep)
        mask = masking(maskduh,image3,Con)
        res = resultmasking(colorgray,maskduh)
        OCRread = ReadEasyOCR(res)
        DigitFinalEasyOCR = FinalResult(OCRread)
        FinalResultImage = FinalEasyOCR(image3,OCRread,SearchCountour)
        st.write(DigitFinalEasyOCR)
        st.image(FinalResultImage)

    with col2:
        st.markdown('<p style="text-align: center;color: white;">Tesseract OCR</p>',unsafe_allow_html=True)
        image2 = np.array(image.convert('RGB'))
        image3 = cv2.cvtColor(image2,1)
        colorgrayTesseract = GrayImageTesseract(image3)
        PrepTesseract = Preprocessing1Tesseract(colorgrayTesseract)
        contourTesseract22 = ContourTesseract2(image3,PrepTesseract)
        countcountoursTesseract1 = countcountoursTesseract(contourTesseract22,image3)
        Tesseractcountour11 = tesseractcontour(image3,countcountoursTesseract1)
        maskingtesseract11 = MaskingTesseract1(colorgrayTesseract)
        maskingtesseract22 = MaskTesseract(maskingtesseract11,countcountoursTesseract1,image3)
        FinalResultImageTesseract = FinalTesseract(maskingtesseract11,colorgrayTesseract)
        st.write(FinalResultImageTesseract)
        st.image(maskingtesseract22)