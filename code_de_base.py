import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import imutils
import matplotlib.pyplot as plt
from PIL import Image
import re
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def orient_vertical(img):
    width = img.shape[1]
    height = img.shape[0]
    if width > height:
        rotated = imutils.rotate(img, angle=270)
    else:
        rotated = img.copy()
    return rotated

def sharpen_edge(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    dilated = cv2.dilate(blurred, rectKernel, iterations=2)
    edged = cv2.Canny(dilated, 75, 200, apertureSize=3)
    return edged

def binarize(img, threshold):
    threshold = np.mean(img)
    thresh, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(binary, rectKernel, iterations=2)
    return dilated

def find_receipt_bounding_box(binary, img):
    global rect
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    largest_cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_cnt)
    box = np.intp(cv2.boxPoints(rect))
    boxed = cv2.drawContours(img.copy(), [box], 0, (0, 255, 0), 20)
    return boxed, largest_cnt

def find_tilt_angle(largest_contour):
    angle = rect[2]
    if angle < -45:
        angle += 90
    uniform_angle = abs(angle)
    return rect, uniform_angle

def adjust_tilt(img, angle):
    if angle >= 5 and angle < 80:
        rotated_angle = 0
    elif angle < 5:
        rotated_angle = angle
    else:
        rotated_angle = 270 + angle
    tilt_adjusted = imutils.rotate(img, rotated_angle)
    delta = 360 - rotated_angle
    return tilt_adjusted, delta

def crop(img, largest_contour):
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped = img[y:y+h, x:x+w]
    return cropped

def enhance_txt(img):
    w = img.shape[1]
    h = img.shape[0]
    w1 = int(w * 0.05)
    w2 = int(w * 0.95)
    h1 = int(h * 0.05)
    h2 = int(h * 0.95)
    ROI = img[h1:h2, w1:w2]
    threshold = np.mean(ROI) * 0.98
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (1, 1), 0)
    edged = 255 - cv2.Canny(blurred, 100, 150, apertureSize=7)
    thresh, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    return binary

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"No image found at {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)
    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    receipt_contour = None
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            receipt_contour = approx
            break
    if receipt_contour is None:
        return image, gray, edged
    receipt = four_point_transform(image, receipt_contour.reshape(4, 2))
    receipt_gray = four_point_transform(gray, receipt_contour.reshape(4, 2))
    return receipt, receipt_gray, edged

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def perform_ocr(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    kernel = np.ones((1, 1), np.uint8)
    gray = cv2.dilate(gray, kernel, iterations=1)
    gray = cv2.erode(gray, kernel, iterations=1)
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(gray, config=custom_config, output_type=Output.STRING)
    return text

def extract_total(text):
    text = re.sub(r'(\d+)\s+(\.\d{2})', r'\1\2', text)
    text = re.sub(r'(\d+)\s*/\s*(\d{2})', r'\1.\2', text)
    
    # Regex patterns
    monetary_pattern = r'\b\d{1,3}(?:,\d{3})*(?:[.,]\d{2})\b'
    total_patterns = [
        re.compile(r'\btotal\b', re.IGNORECASE),
        re.compile(r'\botal\b', re.IGNORECASE),
        re.compile(r'\bamount due\b', re.IGNORECASE),
        re.compile(r'\btotal due\b', re.IGNORECASE),
        re.compile(r'\btotal bue\b', re.IGNORECASE),
        re.compile(r'\bamount bue\b', re.IGNORECASE),
        re.compile(r'\btota\b', re.IGNORECASE),
        re.compile(r'\btot\b', re.IGNORECASE),
        re.compile(r'\bdue\b', re.IGNORECASE),
        re.compile(r'\bdue amount\b', re.IGNORECASE),
        re.compile(r'\bamount\b', re.IGNORECASE),
        re.compile(r'\bamount d\b', re.IGNORECASE),
        re.compile(r'\btotale\b', re.IGNORECASE),
        re.compile(r'\btot\b', re.IGNORECASE),
        re.compile(r'\btotal d\b', re.IGNORECASE),
        re.compile(r'\btotl\b', re.IGNORECASE),
        re.compile(r'\bal\b', re.IGNORECASE),
    ]
    subtotal_pattern = re.compile(r'\bsubtotal\b', re.IGNORECASE)
    
    lines = text.split('\n')
    total_values = []
    other_values = []
    
    for line in lines:
        if any(tp.search(line) for tp in total_patterns) and not subtotal_pattern.search(line):
            matches = re.findall(monetary_pattern, line)
            if matches:
                value = matches[-1].replace(',', '.')
                total_values.append(float(value))
        else:
            matches = re.findall(monetary_pattern, line)
            if matches:
                value = matches[-1].replace(',', '.')
                other_values.append(float(value))
    
    if total_values:
        return max(total_values)
    elif other_values:
        return max(other_values)
    else:
        return None


def preprocess_for_display(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def extract_text_from_receipt(image_path):
    total_values = []
    
    try:
        raw_img = cv2.imread(image_path)
        processed_img = preprocess_for_display(raw_img)
        plt.imshow(processed_img, cmap='gray')
        plt.title("Raw Image")
        plt.show()
        
        rotated = orient_vertical(raw_img)
        processed_rotated = preprocess_for_display(rotated)
        plt.imshow(processed_rotated, cmap='gray')
        plt.title("Oriented Image")
        plt.show()
        
        edged = sharpen_edge(rotated)
        processed_edged = preprocess_for_display(edged)
        plt.imshow(processed_edged, cmap='gray')
        plt.title("Edged Image")
        plt.show()
        
        binary = binarize(edged, 100)
        processed_binary = preprocess_for_display(binary)
        plt.imshow(processed_binary, cmap='gray')
        plt.title("Binarized Image")
        plt.show()
        
        boxed, largest_cnt = find_receipt_bounding_box(binary, rotated)
        processed_boxed = preprocess_for_display(boxed)
        plt.imshow(processed_boxed, cmap='gray')
        plt.title("Bounding Box")
        plt.show()
        
        rect, angle = find_tilt_angle(largest_cnt)
        tilted, delta = adjust_tilt(boxed, angle)
        processed_tilted = preprocess_for_display(tilted)
        plt.imshow(processed_tilted, cmap='gray')
        plt.title(f"Tilt Adjusted Image (Delta: {round(delta, 2)} degrees)")
        plt.show()
        
        cropped = crop(tilted, largest_cnt)
        processed_cropped = preprocess_for_display(cropped)
        plt.imshow(processed_cropped, cmap='gray')
        plt.title("Cropped Image")
        plt.show()
        
        enhanced = enhance_txt(cropped)
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        processed_enhanced = preprocess_for_display(enhanced_rgb)
        plt.imshow(processed_enhanced, cmap='gray')
        plt.title("Enhanced Image")
        plt.show()
        
        ocr_text = perform_ocr(enhanced_rgb)
        total = extract_total(ocr_text)
        if total is not None:
            total_values.append(float(total))
    except Exception as e:
        print(f"First method failed: {e}")
    
    try:
        receipt, receipt_gray, edged = preprocess_image(image_path)
        processed_receipt = preprocess_for_display(receipt)
        plt.imshow(processed_receipt, cmap='gray')
        plt.title("Preprocessed Image")
        plt.show()
        
        ocr_text = perform_ocr(receipt_gray)
        total = extract_total(ocr_text)
        if total is not None:
            total_values.append(float(total))
    except Exception as e:
        print(f"Second method failed: {e}")
    
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        #print(text)
        total = extract_total(text)
        if total is not None:
            total_values.append(float(total))
    except Exception as e:
        print(f"Final method failed: {e}")
    
    if total_values:
        valid_totals = [value for value in total_values if value <= 1000]
        if valid_totals:
            max_total = max(valid_totals)
            print(f"Extracted Total: {max_total}")
            return max_total
        else:
            return "No valid total found under 1000."
    else:
        return "OCR failed."

def process_multiple_images(image_paths):
    results = {}
    for image_path in image_paths:
        print(f"Processing {image_path}...")
        total = extract_text_from_receipt(image_path)
        results[os.path.basename(image_path)] = total
    return results

if __name__ == "__main__":
    image_paths = [
        './data/1132-receipt.jpg',
        './data/1133-receipt.jpg',
        './data/1134-receipt.jpg',
        './data/1135-receipt.jpg',
        './data/1136-receipt.jpg',
        './data/1137-receipt.jpg',
        './data/1138-receipt.jpg',
        './data/1139-receipt.jpg',
        './data/1140-receipt.jpg',
        './data/1141-receipt.jpg',
        './data/1143-receipt.jpg',
        './data/1144-receipt.jpg',
        './data/1145-receipt.jpg',
        './data/1146-receipt.jpg',
        './data/1147-receipt.jpg',
        './data/1148-receipt.jpg',
        './data/1149-receipt.jpg',
        './data/1150-receipt.jpg',
        './data/1151-receipt.jpg',
        './data/1152-receipt.jpg',
        './data/1153-receipt.jpg',
        './data/1154-receipt.jpg',
        './data/1155-receipt.jpg',
        './data/1156-receipt.jpg',
        './data/1157-receipt.jpg',
        './data/1158-receipt.jpg',
        './data/1159-receipt.jpg',
        './data/1160-receipt.jpg',
        './data/1161-receipt.jpg',
        './data/1162-receipt.jpg',
        './data/1163-receipt.jpg',
        './data/1164-receipt.jpg',
        './data/1165-receipt.jpg',
        './data/1166-receipt.jpg',
        './data/1167-receipt.jpg',
        './data/1168-receipt.jpg',
        './data/1169-receipt.jpg',
        './data/1170-receipt.jpg',
        './data/1171-receipt.jpg',
        './data/1175-receipt.jpg',
        './data/1181-receipt.jpg',
        './data/1182-receipt.jpg',
        './data/1183-receipt.jpg',
        './data/1184-receipt.jpg',
        './data/1185-receipt.jpg',
        './data/1188-receipt.jpg',
        './data/1189-receipt.jpg',
        './data/1191-receipt.jpg',
        './data/1192-receipt.jpg',
        './data/1194-receipt.jpg',
        './data/1197-receipt.jpg',
        './data/1198-receipt.jpg'
]
    results = process_multiple_images(image_paths)
    print("Results:", results)
