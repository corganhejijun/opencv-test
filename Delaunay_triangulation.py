import cv2
import os
import dlib

path = 'data/test.jpg'
modelPath = 'models/shape_predictor_68_face_landmarks.dat'

# init model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(modelPath)

# get img landmarks
img = cv2.imread(path)
det = detector(img, 1)[0]
shape = predictor(img, det)

# get delaunay triangles
rect = (0, 0, img.shape[1], img.shape[0])
subdiv = cv2.Subdiv2D(rect)
for i in range(shape.num_parts):
    subdiv.insert((shape.part(i).x, shape.part(i).y))
triangleList = subdiv.getTriangleList()

def insideImage(img, point) :
    if point[0] < 0 :
        return False
    elif point[1] < 0 :
        return False
    elif point[0] > img.shape[1] :
        return False
    elif point[1] > img.shape[0] :
        return False
    return True

def draw_delaunay(img, triangleList, color):
    for t in triangleList:
        pt1 = (t[0], t[1])
        if not insideImage(img, pt1):
            continue
        pt2 = (t[2], t[3])
        if not insideImage(img, pt2):
            continue
        pt3 = (t[4], t[5])
        if not insideImage(img, pt3):
            continue
        cv2.line(img, pt1, pt2, color, 1, cv2.LINE_AA, 0)
        cv2.line(img, pt2, pt3, color, 1, cv2.LINE_AA, 0)
        cv2.line(img, pt3, pt1, color, 1, cv2.LINE_AA, 0)

draw_delaunay(img, triangleList, (255, 255, 255))
cv2.imshow("delaunay trianglation", img)
cv2.waitKey()
