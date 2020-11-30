import cv2
import numpy as np

# load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier("cascade/cascade.xml")


# choose an img to detect faces in
img = cv2.imread("images/girl.jpg")
#img = cv2.imread("images/girl-mask.jpg")
#img = cv2.imread("images/girl-mask-gray.jpg")
#img = cv2.imread("images/girl-mask-gray-sideways.jpg")
#img = cv2.imread("images/body-mask.jpg")
#img = cv2.imread("images/people.jpeg")

# convert to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# draw rectangles 
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 250, 0), 2)
 
print(face_coordinates)

cv2.imshow("Face Detector", img)

cv2.waitKey()

print("Completed") 
