import json
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import requests
import re

# Load SSD model
net = cv2.dnn.readNetFromCaffe("E:/Me_Personal/Punt_Intern_Project/deploy.prototxt", "E:/Me_Personal/Punt_Intern_Project/mobilenet_iter_73000.caffemodel")

# Read the description and image URL from the file
file_path = "image_description.json"

with open(file_path, "r", encoding="utf8") as file:
    data = json.load(file)
    
    if "description" in data and "imageUrl" in data:
        description = data["description"]
        image_url = data["imageUrl"]
    else:
        raise ValueError("The JSON file must contain 'description' and 'imageUrl' keys.")
    
    print("Description from app.js:")
    print(description)

    # Extract the coordinates of the most important object from the description
    important_object_coordinates = None
    match = re.search(r'\((\d+), (\d+)\)', description)
    if match:
        important_object_coordinates = tuple(map(int, match.groups()))
    else:
        raise ValueError("Could not find the coordinates of the most important object in the description")

    # Load the image from URL
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    image_np = np.array(image)

    (h, w) = image_np.shape[:2]

    # Prepare the image for object detection
    blob = cv2.dnn.blobFromImage(cv2.resize(image_np, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Variables to store the most important object's bounding box
    important_startX, important_startY, important_endX, important_endY = None, None, None, None

    # Loop over the detections to process them
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Filter out weak detections
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw bounding box and label for all objects
            label = f"Index: {idx}, Conf: {confidence:.2f}"
            cv2.rectangle(image_np, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image_np, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Check if the current object matches the most important object coordinates
            if important_object_coordinates and (important_object_coordinates[0] >= startX and important_object_coordinates[0] <= endX) and (important_object_coordinates[1] >= startY and important_object_coordinates[1] <= endY):
                important_startX, important_startY, important_endX, important_endY = startX, startY, endX, endY

    # Highlight the most important object if coordinates were found
    if important_startX is not None:
        cv2.rectangle(image_np, (important_startX, important_startY), (important_endX, important_endY), (0, 0, 255), 2)
        y = important_startY - 15 if important_startY - 15 > 15 else important_startY + 15
        cv2.putText(image_np, "Most Important", (important_startX, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        print("No bounding box found for the most important object.")

    # Display the image with bounding boxes
    image_with_boxes = Image.fromarray(image_np)
    image_with_boxes.show()

    # Optionally, save the image with bounding boxes
    image_with_boxes.save("image_with_bounding_boxes.jpg")
