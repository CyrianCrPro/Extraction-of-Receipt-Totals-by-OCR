# pip install streamlit, opencv-python, numpy, pytesseract, scikit-image, pillow

import streamlit as st
import numpy as np
import cv2
from skimage.filters import threshold_local
import pytesseract
import re
from PIL import Image

st.set_page_config(layout="wide")

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # Changer ici le chemin vers tesseract.exe de votre machine


def process_receipt(image):
    def opencv_resize(image, ratio):
        width = int(image.shape[1] * ratio)
        height = int(image.shape[0] * ratio)
        dim = (width, height)
        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    def approximate_contour(contour):
        peri = cv2.arcLength(contour, True)
        return cv2.approxPolyDP(contour, 0.032 * peri, True)

    def get_receipt_contour(contour):
        for c in contour:
            approx = approximate_contour(c)
            if len(approx) == 4:
                return approx
        return None

    def contour_to_rect(contour):
        pts = contour.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect / (500.0 / imopencv.shape[0])

    def wrap_perspective(img, rect):
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    def bw_scanner(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        T = threshold_local(gray, 21, offset=5, method="gaussian")
        return (gray > T).astype("uint8") * 255

    imopencv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    imresized = opencv_resize(imopencv, (500.0 / imopencv.shape[0]))

    col1, spacer1, col2, spacer2, col3, spacer3, col4 = st.columns([3, 0.1, 3, 0.1, 3, 0.1, 3])

    with col1:
        st.write("### Image Redimensionnée")
        st.image(imresized, channels="BGR", use_column_width=True)

    imbw = cv2.cvtColor(imresized, cv2.COLOR_BGR2GRAY)
    imblur = cv2.GaussianBlur(imbw, (3, 3), 0)
    rectkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilation = cv2.dilate(imblur, rectkernel, iterations=1)
    edge = cv2.Canny(dilation, 100, 100, apertureSize=3)

    with col2:
        st.write("### Detection de Contours")
        st.image(edge, channels="GRAY", use_column_width=True)

    contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    imresized_copy = imresized.copy()

    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contour = sorted_contours[0] if sorted_contours else None

    if largest_contour is not None:
        draw_largest_contour = cv2.drawContours(imresized.copy(), [largest_contour], -1, (0, 255, 0), 3)
        with col3:
            st.write("### Plus Gros Contour")
            st.image(draw_largest_contour, channels="BGR", use_column_width=True)
    else:
        return "No contours found.", None

    receipt_contour = get_receipt_contour(sorted_contours)
    if receipt_contour is not None:
        rect = contour_to_rect(receipt_contour)
        warped = wrap_perspective(imopencv, rect)
        scanned = bw_scanner(warped)
        with col4:
            st.write("### Recu Scanné")
            st.image(scanned, channels="GRAY", use_column_width=True)
    else:
        return "Le contour du reçu n'a pas été trouvé.", None

    text = pytesseract.image_to_string(scanned, lang='eng')
    amount_pattern = re.compile(r'(\d+\.\d{2})\s?EUR')
    matches = amount_pattern.finditer(text)
    amounts = [float(match.group(1)) for match in matches if float(match.group(1)) < 10000]

    if amounts:
        total_amount = max(amounts)
        return f"<div style='font-size: 32px;'>Total à régler : {total_amount} EUR</div>", None
    else:
        return "<div style='font-size: 32px;'>Pas de montant valide détecté</div>", None


st.title("Extraction du total d'une facture")
uploaded_file = st.file_uploader("Glissez et déposez votre facture ici", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    result_text, _ = process_receipt(image)
    if result_text:
        st.markdown(result_text, unsafe_allow_html=True)

    # if "Total Amount to Pay" in result_text:
    #     total_amount = result_text.split(": ")[1]
    #     st.write(f"### The Total is: {total_amount} EUR")
    # else:
    #     st.write("### The Total is: --")

# POUR LANCER LE SCRIPT : streamlit run receipt.py dans un terminal à l'endroit du script.