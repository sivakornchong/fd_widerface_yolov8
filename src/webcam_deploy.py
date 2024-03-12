from ultralytics import YOLO
import cv2
import math 
import time
import torch
from torch import nn
from torchvision import transforms
from PIL import Image

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.backends.cudnn.version())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device.type}")

def calcage(input_image, model):
    # Reshape the image to the required input shape for ResNet34
    input_size = (224, 224)
    pil_image = Image.fromarray(input_image)
    resized_image = pil_image.resize(input_size)
    # Define the transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Apply the transformation
    input_image = transform(resized_image)
    input_batch = input_image.unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():  # this stops pytorch doing computational graph stuff under-the-hood and saves memory and time 
        outputs = model(input_batch)
        y_hat = nn.functional.softmax(outputs, dim=1)  # This is to calculate the losses. 
        y_hat_labels = torch.argmax(y_hat, dim=1)
        # print(y_hat)
        # print("The frame has an image with:", labels[y_hat_labels])
    return y_hat_labels

# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Video writer setup
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

# Test with an self-trained face model based on YOLOV8, using face detection dataset
model = YOLO("model/best.pt")

model_age_load_path = 'model/fd_age_cls_190224_2.pt'
model_age = torch.load(model_age_load_path).to(device)
model_age.eval()

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
labels = ['Young Children','Children and Youth','Young Adults', 'Adults', 'Older Adults']

save = False
age = True
video = True
expand_ratio = 0.2

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

            # Calculate expanded coordinates
            x1 = max(0, int(x1 - expand_ratio * (x2 - x1)))
            y1 = max(0, int(y1 - expand_ratio * (y2 - y1)))
            x2 = min(img.shape[1], int(x2 + expand_ratio * (x2 - x1)))
            y2 = min(img.shape[0], int(y2 + expand_ratio * (y2 - y1)))


            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            # print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            # print("Class name -->", classNames[cls])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            # cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

            # Check if the detected object is a "person"
            save_directory = "imgs/"


            if classNames[cls] == "face":
                # Crop the person from the image
                cropped_person = img[y1:y2, x1:x2]

                if save:
                    # Save the cropped person image
                    filename = f"{save_directory}cropped_person_{time.time()}.jpg"
                    cv2.imwrite(filename, cropped_person)

                    # Increment the counter for the next person
                    person_counter += 1

                if age:
                    label_index = calcage(cropped_person, model_age)   
                    cv2.putText(img, labels[label_index], org, font, fontScale, color, thickness)

                else:
                    cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

                if video:
                    # Write the frame to the video output
                     out.write(img)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break
    
    time.sleep(0.1)  # Introduce a delay of 1 second

# Release the video writer and capture objects
out.release()
cap.release()
cv2.destroyAllWindows()

# Source code has been adapted from https://dipankarmedh1.medium.com/real-time-object-detection-with-yolo-and-webcam-enhancing-your-computer-vision-skills-861b97c78993