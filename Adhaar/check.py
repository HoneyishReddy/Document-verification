from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import easygui
from sklearn.metrics.pairwise import cosine_similarity
import pytesseract
import re

# Specify the Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'  # Adjust this path accordingly

def get_detected_objects(image, model, threshold=0.4):
    results = model(image)[0]
    detected_objects = []

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            detected_objects.append({
                'class': results.names[int(class_id)],
                'confidence': score,
                'relative_coordinates': {
                    'x1': x1 / image.shape[1],
                    'y1': y1 / image.shape[0],
                    'x2': x2 / image.shape[1],
                    'y2': y2 / image.shape[0]
                }
            })

    return detected_objects

def normalize_bbox(bbox, image_width, image_height):
    return [coord / image_width if i % 2 == 0 else coord / image_height for i, coord in enumerate(bbox)]

def bbox_cosine_similarity(bbox1, bbox2):
    return cosine_similarity([bbox1], [bbox2])[0][0]

def are_objects_similar(user_obj, template_obj):
    class_similarity_threshold = 0.8
    position_similarity_threshold = 0.9

    class_similarity = 1 if user_obj['class'] == template_obj['class'] else 0

    user_normalized_bbox = normalize_bbox(user_obj['relative_coordinates'].values(), user_input_image.shape[1], user_input_image.shape[0])
    template_normalized_bbox = normalize_bbox(template_obj['relative_coordinates'].values(), template_image.shape[1], template_image.shape[0])
    position_similarity = bbox_cosine_similarity(user_normalized_bbox, template_normalized_bbox)

    return class_similarity > class_similarity_threshold and position_similarity > position_similarity_threshold

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

def is_adhaar_number(text):
    # Using a regular expression to match the pattern 'ABCDE1234F'
    adhaar_pattern = r'^\d{12}$'
    return re.match(adhaar_pattern, text) is not None

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

# Load the user-input image
user_input_path = 'Adhaar/bgremoved.jpg'
user_input_image = cv2.imread(user_input_path)

# Load the template image
template_path = 'Adhaar/adhaar.jpg'
template_image = cv2.imread(template_path)

# Load a YOLOv5 model
model_path = 'Adhaar/adhaar.pt'
model = YOLO(model_path)

# Get detected objects for user-input image and template image
user_input_objects = get_detected_objects(user_input_image, model)
template_objects = get_detected_objects(template_image, model)

# Compare detected objects
similar_objects_count = 0

for user_obj in user_input_objects:
    for template_obj in template_objects:
        if are_objects_similar(user_obj, template_obj):
            print(f"Similar object detected: {user_obj['class']} at relative coordinates {user_obj['relative_coordinates']}")
            similar_objects_count += 1

# Save the output image with bounding boxes for user input
for user_obj in user_input_objects:
    x1, y1, x2, y2 = (
        int(user_obj['relative_coordinates']['x1'] * user_input_image.shape[1]),
        int(user_obj['relative_coordinates']['y1'] * user_input_image.shape[0]),
        int(user_obj['relative_coordinates']['x2'] * user_input_image.shape[1]),
        int(user_obj['relative_coordinates']['y2'] * user_input_image.shape[0])
    )
    cv2.rectangle(user_input_image, (x1, y1), (x2, y2), (0, 255, 0), 4)

# Display the final user-input image if more than half of the classes are similar
if similar_objects_count > len(user_input_objects) / 2:
    # Extract text and draw boxes
    user_input_image, text_data = extract_text_and_draw_boxes(user_input_image)

    # Check if PAN number is detected
    for i in range(len(text_data['text'])):
        text = text_data['text'][i]
        if is_adhaar_number(text):
            print(f"Adhhar Number Detected: {text}")

    # Display the result with text and bounding boxes
    display_text_with_boxes(user_input_image, text_data)
    print("Adhaar Card Detected")
else:
    print("Not a Adhaar Card")
 