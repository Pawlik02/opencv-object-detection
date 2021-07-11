import cv2 as cv
import os
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

cascade = []
objects_number = 2
names = os.listdir('haar/')
colors = []

for i in range(objects_number):
    cascade.append(cv.CascadeClassifier(f'/home/pawel/Documents/programy/opencv/detection/haar/{names[i]}/cascade/cascade.xml'))
    colors.append(get_random_color())

video = cv.VideoCapture('/home/pawel/Documents/programy/opencv/detection/test_film.mp4')

while True:
    isTrue, frame = video.read()

    # Detection
    rectangles = []
    if_appears = True
    for i in range(objects_number):
        rectangles.append(cascade[i].detectMultiScale(frame))

    # Draw
    for i in range(objects_number):
        for (x,y,w,h) in rectangles[i]:
            cv.rectangle(frame, (x, y), (x + w, y + h), colors[i], thickness=2)
            cv.putText(frame, names[i], (x, y - 10), cv.FONT_HERSHEY_PLAIN, 3, colors[i], 3)
            for rectangle in rectangles:        
                if type(rectangle) == tuple:
                    if_appears = False
            if if_appears == True:
                # Indicator
                cv.circle(frame, (1800, 1000), 40, (0, 0, 255), -1)
                
    # Displaying frames
    frame = rescale_frame(frame, 0.5)
    cv.imshow('Object detection', frame)
    
    if cv.waitKey(20) and ord('d')==0xFF:
        break

video.release()
cv.destroyAllWindows()