# Face Detection and Age Classification Demonstration

### Purpose
In this scenario, we assume there is an application from health promotion board that would propose different health exercises to different age groups. Hence, a face detection and an age classification algorithm would be very helpful for determining the user's age.

Age range definition based on Health Promotion Board of Singapore
- (A0) Young Children: 0-6 years 
- (A1) Children and Youth: 7-17 years 
- (A2) Youth Adults: 18-25 years
- (A3) Adults: 26-49 years
- (A4) Older adults: 50 years

Disclaimer: This is a model training demonstration for educational purposes only. 

### Algorithm and Training Data
#### Face Detection Model:
- YOLOV8 base architecture
- Finetuned based on WiderFace dataset:
Link: http://shuoyang1213.me/WIDERFACE/
- Eventual MAP on test dataset is:

#### Age Classification Model:
- RESNET34 base architecture
- Finetuned based on UTKFace dataset. The dataset is crtopped using the engine from face detection model. 
Link: https://susanqq.github.io/UTKFace/
- Model is trained up to epoch 4, which is selected to prevent overfitting on training data
- Weighted average accuracy is 74% on the test dataset.

### Deployment
- A simple .py script is developed to deploy both models in a sample demo.
  - The face detection model is used to detect a face. That face is cropped as an image for further processing. 
  - The classification model is used for each cropped image.
  - The results are then demonstrated on screen.
- Demo
  
![demo_gif](https://github.com/sivakornchong/fd_widerface_yolov8/assets/104509481/38451ed4-b0bc-42ae-b804-aac92683b6e9)

### Potential Improvement
- Improve variability within the dataset used for training both models.
  - For classification model specifically, data preprocessing could be done to rotate the faces around and improve the accuracy upon deploymnent.
- Model quantization to reduce model sizes for deployment on smaller devices (phones). 
