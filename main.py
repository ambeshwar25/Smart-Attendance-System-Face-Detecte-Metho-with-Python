import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime

path = 'ImagesAttendance'
images = []
classNames = []
studentIDs = []

# 1- Load Image
for cl in os.listdir(path):
    curImg = cv2.imread(f'{path}/{cl}')
    if curImg is None:
        continue

    images.append(curImg)

    fileName = os.path.splitext(cl)[0]   # e.g. 101_rahul
    parts = fileName.split('_')

    if len(parts) >= 2:
        studentIDs.append(parts[0])
        classNames.append(parts[1])
    else:
        studentIDs.append("Unknown")
        classNames.append(fileName)

# 2- Encodee
def findEncodings(images):
    encodeList = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)

        if len(encodes) > 0:
            encodeList.append(encodes[0])
        else:
            print(" Face not detected in one image")

    return encodeList

# 3- Attendance
def markAttendance(name, student_id):
    with open('Attendance.csv', 'a+') as f:
        f.seek(0)
        data = f.readlines()
        nameList = []

        for line in data:
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{student_id},{dtString}')

# 4- Main
encodeListKnown = findEncodings(images)
print(" Encoding Complete")

total_students = len(classNames)
print(f"Total Students: {total_students}")

present_students = set()

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):

        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        if len(faceDis) > 0:
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                student_id = studentIDs[matchIndex]

                y1,x2,y2,x1 = faceLoc
                y1,x2,y2,x1 = y1*4, x2*4, y2*4, x1*4

                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(img,f'{name} ({student_id})',(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

                markAttendance(name, student_id)
                present_students.add(name)

    # 5- Display Count
    cv2.putText(img, f'Total: {total_students}', (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.putText(img, f'Present: {len(present_students)}', (10,70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow('Smart Attendance System', img)

    if cv2.waitKey(1) == 13:  # Enter key to exit
        break

cap.release()
cv2.destroyAllWindows()