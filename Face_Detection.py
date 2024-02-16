import cv2


def face_location():
    #Loading the Haarcasecade file
    face = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

    #Readingin the image
    #image=cv2.imread("test.jpg")

    vid = cv2.VideoCapture(0)

    while (True):
        ret, frame = vid.read()
        # Convertion of Gray color image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detecting the facial features
        faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
        # Drawing the Rectangle box surrounded at the detected face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)
        #cv2.imwrite("face.jpg", frame)

        # Dsiplay the face detection of output image
        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object

    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


face_location()