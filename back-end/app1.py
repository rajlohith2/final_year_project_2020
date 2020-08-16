import multiprocessing
import pymongo
from pymongo import MongoClient
from flask import Flask, request, jsonify
import cv2
import numpy as np
from PIL import Image
import os
import json
from twilio.rest import Client
import smtplib, ssl
from flask_cors import CORS
import psutil
import subprocess
import time
app = Flask(__name__)
CORS(app)
sender_email = "rajlohith2@gmail.com"
receiver_email = "bharadwajkarthik7@gmail.com"
port = 465  # For SSL
password = "virendersehwag"
cameraIPList = [0]
jobList = []
context = ssl.create_default_context()
currentNewUser = 0
crim_list = {}


def trainData():
    path = 'dataset'
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces,ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))

    # Save the model into trainer/trainer.yml
    recognizer.write('trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi

    # Print the numer of faces trained and end program
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))



def recordCriminalFaceData(filename, cameraIP, face_id):
    if filename == " ":
        cam = cv2.VideoCapture('http://192.168.0.100:8080/video?type=some.mjpeg')
    else:
        cam = cv2.VideoCapture('http://192.168.0.100:8080/video?type=some.mjpeg')

    cam.set(3, 640) # set video width
    cam.set(4, 480) # set video height

    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    print("\n [INFO] Initializing face capture. Look the camera and wait ...")
    # Initialize individual sampling face count
    count = 0

    while(True):
        time.sleep(0.1)
        ret, img = cam.read()
        #img = cv2.flip(img, -1) # flip video image vertically
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
            count += 1

            # Save the captured image into the datasets folder
            cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        if filename == " ":
            if count >= 60:
                break

    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()

def performPredictionFromVideo(filepath):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)
    #iniciate id counter
    id = 0

    # names related to ids: example ==> Marcelo: id=1,  etc
    names = ['None', 'veer', 'Lohith', 'Ganapati', 'Z', 'W']
    cam = cv2.VideoCapture('http://192.168.0.100:8080/video?type=some.mjpeg')
    # Initialize and start realtime video capture
    cam.set(3, 640) # set video widht
    cam.set(4, 480) # set video height

    # Define min window size to be recognized as a face
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)

    while True:
        
        ret, img =cam.read()
        #img = cv2.flip(img, -1) # Flip vertically
        try:
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        except:
                return(crim_list)

        faces = faceCascade.detectMultiScale( 
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
        )

        for(x,y,w,h) in faces:

            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

            # Check if confidence is less them 100 ==> "0" is perfect match 
            if (confidence < 100):
                confidence = int(round(100 - confidence))
                if int(confidence) >= 50:
                    print(id)
                    print(confidence)
                    if id in c_list:
                        if crim_list[id]<confidence:
                            crim_list[id]=confidence
                            print(crim_list[id])
                    else:
                        crim_list[id]=confidence
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
            
    print("\n--------List of Criminals Identified--------\n")
    for name,val in c_list.items():
        print(name,":-Identified with",val,"accuracy")
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    return(crim_list)

def performPrediction(filename, cameraIP, c_list, pid):
    process_id = os.getpid()
    pid[0] = process_id
    print(pid[0])

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)
    #iniciate id counter
    id = 0

    # names related to ids: example ==> Marcelo: id=1,  etc
    names = ['None', 'Lohith', 'Lohith', 'Ganapati', 'Z', 'W']
    if filename == " ":
        cam = cv2.VideoCapture('http://192.168.0.100:8080/video?type=some.mjpeg')
    else:
        cam = cv2.VideoCapture('http://192.168.0.100:8080/video?type=some.mjpeg')
    # Initialize and start realtime video capture
    cam.set(3, 640) # set video widht
    cam.set(4, 480) # set video height

    # Define min window size to be recognized as a face
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)

    while True:
        time.sleep(1)
        ret, img =cam.read()
        #img = cv2.flip(img, -1) # Flip vertically

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale( 
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
        )

        for(x,y,w,h) in faces:

            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

            # Check if confidence is less them 100 ==> "0" is perfect match 
            if (confidence < 100):
                confidence = int(round(100 - confidence))
                if int(confidence) >= 25:
                    print(id)
                    print(confidence)
                    if id in c_list:
                        if c_list[id]<confidence:
                            c_list[id]=confidence
                            print(c_list[id])
                    else:
                        c_list[id]=confidence
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
            
    print("\n--------List of Criminals Identified--------\n")
    for name,val in c_list.items():
        print(name,":-Identified with",val,"accuracy")
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
# function to get the images and label data

def getImagesAndLabels(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')
        detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids

@app.route('/block-camera')
def block():
    os.popen('TASKKILL /PID '+str(pid[0])+' /F 2>NUL')

@app.route('/release-camera')
def release():
    p = multiprocessing.Process(target = performPrediction,args = (" ",0, c_list, pid))
    p.start()

@app.route('/recording')
def recordRealTime():
    os.popen('TASKKILL /PID '+str(pid[0])+' /F 2>NUL')
    filter = {} 
    collection = db["trial_new"]
    doc_count = collection.count_documents(filter)
    currentNewUser = doc_count + 1
    recordCriminalFaceData(" ", 0, currentNewUser)
    p = multiprocessing.Process(target = performPrediction,args = (" ",0, c_list, pid))
    p.start()
    p2 = multiprocessing.Process(target = trainData)
    p2.start()
    return jsonify({})

@app.route('/recording-from-footage')
def recordingFromFootage():
    face_id = request.args['userid']
    recordCriminalFaceData("C:\\Users\\rajlo\\proj\\Final_year_project\\uploads\\sample.mp4",0, face_id)


@app.route('/recognize-from-footage')
def recognizeFromFootage():
    image_path = request.args['image-path']
    print(image_path)
    searchResult = performPredictionFromVideo("C:\\Users\\rajlo\\proj\\Final_year_project\\uploads\\"+image_path)
    collection = db["trial_new"]
    detectedList = []
    for x in searchResult.keys():
        print(x)
        res = collection.find_one({"ID" : int(x)},{ "_id": 0 })
        detectedList.append(res)
    print(detectedList)    
    return jsonify(detectedList)
    

@app.route('/recognition')
def recognizeRealTime():
    print(c_list)
    message = "The following criminals have been discovered in your area at camera 0: \n"
    for i in c_list.keys():
        message += str(i) + "\n"
    criminals_detected = json.dumps(c_list.copy())
    client = Client("AC005dc1deb4ae1c3efd6265aeacf69997", "c4a9eb554443c3a39b8d8ba673f5a3a4")
    client.messages.create(to="+918553587952", 
                       from_="+12058439615", 
                       body=message)
    with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
        server.login("rajlohith2@gmail.com", password)
        # TODO: Send email here
        server.sendmail(sender_email, receiver_email, message)
        collection = db["trial_new"]
    detectedList = []
    for x in c_list.keys():
        print(x)
        res = collection.find_one({"ID" : int(x)},{ "_id": 0 })
        detectedList.append(res)
    print(detectedList)    
    c_list.clear()
    return jsonify(detectedList)

@app.route('/search-by-id')
def searchById():
    detectedList = []
    id = request.args['id']
    collection = db["trial_new"]
    res = collection.find_one({"ID" : int(id)},{ "_id": 0 })
    detectedList.append(res)
    return jsonify(detectedList)

@app.route('/search-by-name')
def searchByName():
    detectedList = []
    fn = request.args['fn']
    ln = request.args['ln']
    collection = db["trial_new"]
    res = collection.find_one({"First_Name" : fn, "Last_Name" : ln},{ "_id": 0 })
    detectedList.append(res)
    return jsonify(detectedList)

@app.route('/admin-list')
def adminList():
    admins = []
    collection = db["admin_table"]
    cursor = collection.find({},{ "_id": 0 })
    print(cursor)
    for document in cursor:
          print(document)
          admins.append(document)
    return jsonify(admins)


@app.route('/add-criminal',methods = ['POST'])
def addCriminal():
    # criminalDetails = {
    #        "First_Name" : request.form["First_Name"],
    #        "Last_Name" : request.form["Last_Name"],
    #        "Gender" : request.form["Gender"],
    #        "Date_Of_Birth" : request.form["Date_Of_Birth"],
    #        "Aadhar_No" : request.form["aadhaar"], 
    #        "Place" : request.form["Place"],
    #        "City" : request.form["City"],
    #        "country" : request.form["country"],
    #        "pincode" : request.form["pincode"],
    #        "Crimes" : request.form["Crimes"],
    #        "Image" : request.form["Image"]
    # }
    data = request.get_data()
    filter = {} 
    collection = db["trial_new"]
    doc_count = collection.count_documents(filter)
    currentNewUser = doc_count + 1
    print("DATA ::::::",data)
    data = json.loads(data)
    data.update({"ID" : currentNewUser})
    x = collection.insert_one(data)
    print(x.inserted_id)
    return jsonify({})

@app.route('/login')
def login():
    user = request.args['user']
    password = request.args['pass']
    print(user)
    print(password)
    collection = db["admin_table"]
    res = collection.find_one({"User_id" : user, "password" : password},{ "_id": 0 })
    print(res)
    if res== None:
        return jsonify({"message":"failure"})
    return jsonify({"message":"success"})


@app.route('/add-user')
def addUser():
    user = request.args['user']
    password = request.args['pass']
    print(user)
    print(password)
    collection = db["admin_table"]
    doc_count = collection.count_documents({})
    currentNewUser = doc_count + 1
    res = collection.insert_one({"ID": currentNewUser,"User_id" : user, "password" : password})
    print(res)
    return jsonify({"message":"success"})
    
@app.route('/remove-user')
def removeUser():
    user = request.args['user']
    password = request.args['pass']
    print(user)
    print(password)
    collection = db["admin_table"]
    res = collection.delete_one({"User_id" : user, "password" : password})
    print(res)
    return jsonify({"message":"success"})




if __name__ == '__main__':
    manager = multiprocessing.Manager()
    c_list=manager.dict()
    pid = manager.list([0])
    path = 'dataset'

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces,ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))

    # Save the model into trainer/trainer.yml
    recognizer.write('trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi

    # Print the numer of faces trained and end program
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
    p = multiprocessing.Process(target = performPrediction,args = (" ",0, c_list, pid))
    p.start()
    client = MongoClient('mongodb+srv://rajlohith2:bit123@criminal-database-drxaw.mongodb.net/test?retryWrites=true&w=majority')
    db = client.CriminalDB
    app.run()