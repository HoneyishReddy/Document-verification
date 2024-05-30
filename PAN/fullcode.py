import cv2
import numpy as np
import matplotlib.pyplot as plt
from rembg import remove
from PIL import Image
import easygui
from skimage.metrics import structural_similarity as ssim
import pytesseract
import os
from ultralytics import YOLO
import re

def calculate_brightness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    average_brightness = np.mean(gray)
    return average_brightness

def plot_grayscale_histogram(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    plt.plot(hist)
    plt.title('Grayscale Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.show()

def calculate_hsd_feature(input_image, ground_truth_image, num_bins):
    input_hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    
    # Resize the ground truth image to match the dimensions of the input image
    ground_truth_resized = cv2.resize(ground_truth_image, (input_image.shape[1], input_image.shape[0]))

    ground_truth_hsv = cv2.cvtColor(ground_truth_resized, cv2.COLOR_BGR2HSV)

    # Split the HSV channels
    input_h, input_s, _ = cv2.split(input_hsv)
    ground_truth_h, ground_truth_s, _ = cv2.split(ground_truth_hsv)

    # Calculate hue and saturation differences
    hue_diff = np.abs(input_h - ground_truth_h)
    saturation_diff = np.abs(input_s - ground_truth_s)

    # Calculate the sum of differences in each rectangular bin
    h_bins = np.linspace(0, 180, num_bins + 1)
    s_bins = np.linspace(0, 255, num_bins + 1)

    h_bin_indices = np.digitize(input_h, h_bins) - 1
    s_bin_indices = np.digitize(input_s, s_bins) - 1

    feature_vector = np.zeros((num_bins, num_bins), dtype=np.float32)

    for i in range(num_bins):
        for j in range(num_bins):
            indices = np.where((h_bin_indices == i) & (s_bin_indices == j))
            feature_vector[i, j] = np.sum(hue_diff[indices]) + np.sum(saturation_diff[indices])

    return feature_vector

def remove_background_rembg(image_path):
    input_image = Image.open(image_path)
    output_image = remove(input_image)
    output_np = np.array(output_image)

    if output_np.shape[2] == 4 and np.all(output_np[:, :, 3] == 255):
        return cv2.imread(image_path)
    else:
        return output_np

def apply_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area to exclude small contours
    min_contour_area = 1000
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    # Approximate contours to reduce the number of points
    epsilon = 0.02 * cv2.arcLength(contours[0], True)
    approx = cv2.approxPolyDP(contours[0], epsilon, True)

    # Create an empty mask
    mask = np.zeros_like(gray)

    # Draw contours on the mask
    cv2.drawContours(mask, [approx], -1, (255), thickness=cv2.FILLED)

    # Apply the mask to the original image
    result_with_contours = cv2.bitwise_and(image, image, mask=mask)

    return result_with_contours, [approx]

def display_hue_saturation_images(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, _ = cv2.split(hsv_image)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1), plt.imshow(hue, cmap='hsv'), plt.title('Hue')
    plt.subplot(1, 3, 2), plt.imshow(saturation, cmap='hsv'), plt.title('Saturation')
    plt.subplot(1, 3, 3), plt.imshow(image), plt.title('Original Image with Contours')

    plt.show()

def extract_text_and_draw_boxes(image):
    # Use pytesseract to perform OCR and get detailed information
    data = pytesseract.image_to_data(image, config='--psm 6', output_type=pytesseract.Output.DICT)

    for i in range(len(data['text'])):
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            confidence = int(data['conf'][i])
            text = data['text'][i]

            # Filter out low-confidence detections and non-rectangular bounding boxes
            if confidence > 60 and w > 5 and h > 5 and w / h > 0.2:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                print(f"Text: {text}, Confidence: {confidence}, Bounding Box: ({x}, {y}, {x + w}, {y + h})")
    return image, data

def is_pan_number(text):
    # Using a regular expression to match the pattern 'ABCDE1234F'
    pan_pattern = r'^[A-Z]{5}\d{4}[A-Z]$'
    return re.match(pan_pattern, text) is not None

def display_text_with_boxes(image, data):
    for i in range(len(data['text'])):
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        confidence = int(data['conf'][i])
        text = data['text'][i]

        # Filter out low-confidence detections and non-rectangular bounding boxes
        if confidence > 60 and w > 5 and h > 5 and w / h > 0.2:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f"{text} ({confidence}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Result with Text and Bounding Boxes')
    plt.show()

# Select the input image file using a file dialog
input_path = easygui.fileopenbox(title='Select image file')
ground_truth_path = easygui.fileopenbox(title='Select ground truth image file')

# Remove background using rembg
result_rembg = remove_background_rembg(input_path)

# Calculate brightness of the original and processed images
brightness_original = calculate_brightness(cv2.imread(input_path))
brightness_processed = calculate_brightness(result_rembg)

# Plot the grayscale histogram
plot_grayscale_histogram(result_rembg)

# Apply contours to the processed image
result_with_contours, contours = apply_contours(result_rembg)

# Display brightness information
print(f"Original Image Brightness: {brightness_original}")
print(f"Processed Image Brightness: {brightness_processed}")

# Display Hue and Saturation images
display_hue_saturation_images(result_rembg)

# Display the original image
plt.figure(figsize=(5, 5))
plt.imshow(cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.show()

# Display the background-removed image
plt.figure(figsize=(5, 5))
plt.imshow(cv2.cvtColor(result_rembg, cv2.COLOR_BGR2RGB))
plt.title('Background Removed')
plt.show()

# Display the result with contours
plt.figure(figsize=(5, 5))
plt.imshow(result_with_contours)
plt.title('Result with Contours')
plt.show()

# Extract text and draw bounding boxes
result_with_text, text_data = extract_text_and_draw_boxes(result_rembg)

# Display result with text and bounding boxes
display_text_with_boxes(result_with_text, text_data)

# Display contours only
plt.figure(figsize=(5, 5))
plt.imshow(cv2.drawContours(result_rembg.copy(), contours, -1, (0, 255, 0), 5))
plt.title('Contours Only')
plt.show()

# Calculate structural similarity score
gray_original = cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2GRAY)
gray_processed = cv2.cvtColor(result_rembg, cv2.COLOR_BGR2GRAY)
score, _ = ssim(gray_original, gray_processed, full=True)
print(f'Structural Similarity Score: {score:.4f}')

# Calculate HSD feature vector
num_bins = 8
feature_vector = calculate_hsd_feature(result_rembg, cv2.imread(ground_truth_path), num_bins)
print('HSD Feature Vector:')
print(feature_vector)

# YOLOv5 Detection
image_path = 'image/cards (3).jpg'
OUTPUT_DIR = os.path.join('.', 'validates')
output_image_path = os.path.join(OUTPUT_DIR, 'output.jpg')

# Load an image
oframe = cv2.imread(image_path)
frame = cv2.resize(oframe, (420, 640))
H, W, _ = frame.shape

model_path = 'validate\\best (5).pt'

# Load a model
model = YOLO(model_path)

threshold = 0.4

results = model(frame)[0]

# List to track detected classes
detected_classes = []
print(results.boxes.data.tolist())
for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result

    if score > threshold:
        detected_classes.append(results.names[int(class_id)])
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
        cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        # Print information for each detected box
        print("Detected:", results.names[int(class_id)], " with confidence:", score)

# Save the output image
cv2.imwrite(output_image_path, frame)

# Display the final YOLOv5 result
plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.title('YOLOv5 Result')
plt.show()