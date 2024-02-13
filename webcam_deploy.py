from ultralytics import YOLO
import cv2
import math 
import time

# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Model
# model = YOLO("yolo-Weights/yolov8n.pt")

# Test with an self-trained face model based on YOLOV8, using face detection dataset
model = YOLO("model/best.pt")


# object classes
# classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
#               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
#               "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
#               "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
#               "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
#               "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
#               "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
#               "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
#               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
#               "teddy bear", "hair drier", "toothbrush"
#               ]

classNames = ["face"]

# Counter for unique filenames
person_counter = 1

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

            # Check if the detected object is a "person"
            save_directory = "imgs/"

            if classNames[cls] == "face":
                # Crop the person from the image
                cropped_person = img[y1:y2, x1:x2]

                # Save the cropped person image
                filename = f"{save_directory}cropped_person_{time.time()}.jpg"
                cv2.imwrite(filename, cropped_person)

                # Increment the counter for the next person
                person_counter += 1


    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break
    
    time.sleep(1)  # Introduce a delay of 1 second

cap.release()
cv2.destroyAllWindows()

# Source code has been adapted from https://dipankarmedh1.medium.com/real-time-object-detection-with-yolo-and-webcam-enhancing-your-computer-vision-skills-861b97c78993