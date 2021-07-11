import cv2 as cv
import numpy as np
import random

# Custom functions
def rescale_frame(frame, scale):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)
    
def get_random_color():
    b = random.randrange(0, 255)
    g = random.randrange(0, 255)
    r = random.randrange(0, 255)
    return (b, g, r)

classes = []
colors = []
file = open('/home/pawel/Documents/programy/opencv/detection/yolo/classes.txt', 'r')
data = file.readlines()
for i in range(len(data)):
    classes.append(data[i].strip())
    colors.append(get_random_color())

file.close()

network = cv.dnn.readNet('/home/pawel/Documents/programy/opencv/detection/yolo/yolov3_training_last.weights', '/home/pawel/Documents/programy/opencv/detection/yolo/yolov3_testing.cfg')
video = cv.VideoCapture('/home/pawel/Documents/programy/opencv/detection/test_film.mp4')

while True:
    isTrue, frame = video.read()
    height, width, _ = frame.shape

    blob = cv.dnn.blobFromImage(frame, 1/255, (416, 416), (0, 0, 0), crop=False)
    network.setInput(blob)
    output_layers_names = network.getUnconnectedOutLayersNames()
    layerOutputs = network.forward(output_layers_names)

    rectangles = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.1:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                rectangles.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv.dnn.NMSBoxes(rectangles, confidences, 0.1, 0.4)
    
    for i in range(len(rectangles)):
        if i in indexes:
            x, y, w, h = rectangles[i]
            cv.rectangle(frame, (x, y), (x + w, y + h), colors[class_ids[i]], thickness=2)
            cv.putText(frame, classes[class_ids[i]], (x, y - 10), cv.FONT_HERSHEY_PLAIN, 3, colors[class_ids[i]], 3)
    
    # Displaying frames
    frame = rescale_frame(frame, 0.5)
    cv.imshow('Object detection', frame)
    
    if cv.waitKey(20) and ord('d')==0xFF:
        break

cv.destroyAllWindows()