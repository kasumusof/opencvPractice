import cv2 as cv
import numpy as np

# To reading the video file
cap = cv.VideoCapture('vtest.avi')

_, frame1 = cap.read()
_, frame2 = cap.read()

while cap.isOpened():

    diff = cv.absdiff(frame1, frame2)

    gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
    g_b = cv.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv.threshold(g_b, 20, 255, cv.THRESH_BINARY)
    dilated = cv.dilate(thresh, None, iterations=3)
    contours, _ = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x, y, w, h) = cv.boundingRect(contour)

        if cv.contourArea(contour) < 900:
            continue  # This would do nothing if the contour area is less than 900
        cv.rectangle(frame1, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # To add a text when motion is detected.
        cv.putText(frame1, 'Status: movement', (10, 20), cv.FONT_HERSHEY_SIMPLEX,
                   1, (255, 0, 0), 3)

    cv.imshow('VideoFeed', frame1)

    frame1 = frame2
    ret, frame2 = cap.read()

    if cv.waitKey(40) == 27:
        break

cap.release()
cv.destroyAllWindows()
