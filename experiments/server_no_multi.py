
from ast import YieldFrom
from tokenize import Name
import cv2  
import numpy as np
import websockets
import asyncio
import struct, pickle, base64
import time
from insightface.app import FaceAnalysis
import random
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore, storage
import os
import atexit
import getmac
# from emb_data import db, arcface, bucket
import threading
from inference import get_model, inference
import glob
import logging
logging.basicConfig(level = logging.CRITICAL)
cred = credentials.Certificate('mykey.json')
firebase_admin.initialize_app(cred, {
     'databaseURL' : 'https://arface-79a8a-default-rtdb.firebaseio.com/',
     'storageBucket' : 'arface-79a8a.appspot.com'
})

db = firestore.client()
bucket = storage.bucket()

arcface = FaceAnalysis(name = 'buffalo_sc', providers = ['CPUExecutionProvider'])
arcface.prepare(ctx_id=0, det_size=(400, 400))
receiveName = False
name = None
isUploading = False
model = get_model()

def close_camera():
   global db
   macAddress = getmac.get_mac_address()
   db.collection('camera_list').document(macAddress).update({'state' : 'Unavailable'})

def fileUpload(fname, name, hash):
  global bucket

  blob = bucket.blob('registered_person/' + name + '/' + hash)
  blob.upload_from_filename(fname)
  os.remove(fname)
  # os.rmdir(fname[:fname.rfind('/')])
  blob.make_public()
  return blob.public_url

def makeLocalImageAndUpload(name, frame):
    file_dir = 'static/' + name
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    hash = "%032x" % random.getrandbits(128)
    file_path = os.path.join(file_dir, hash+ '.jpg')
    cv2.imwrite(file_path, frame)
    return fileUpload(file_path, name, hash)

def updateFirebase(frame, name):
    global arcface, db
    face_dict_list = arcface.get(frame)
    croppedFrame = makeCroppedFace(frame)

    if len(face_dict_list) > 0:
        face = face_dict_list[0]

        if face['det_score'] > 0.5:    
            emb = list(face['embedding'].astype(float))
            if not db.collection('embedding_name').document(name).get().exists:
                url = makeLocalImageAndUpload(name, croppedFrame)
                db.collection('embedding_name').document(name).set({
                    'name': name,
                    'embedding_vectors': [{'embedding_vector' : emb}],
                    'count': 1,
                    'image_paths' : [{'image_path' : url}]
                    })
               
            # 기존에 등록된 name을 추가로 등록하는 경우
            # else:
            #     before_count = len(db.collection('embedding_name').document(name).get().to_dict()['embedding_vectors'])
            #     db.collection('embedding_name').document(name).update({'embedding_vectors': firestore.ArrayUnion([{'embedding_vector' : emb}])})
            #     after_count = len(db.collection('embedding_name').document(name).get().to_dict()['embedding_vectors'])
            # # 만약 같은 이미지가 등록되어 있으면 업데이트를 하지않는다.
            #     if before_count != after_count:
            #         url = makeLocalImageAndUpload(name, croppedFrame)
            #         db.collection('embedding_name').document(name).update({
            #             'count' : after_count,
            #             'image_paths' : firestore.ArrayUnion([{'image_path' : url}])})
            else:
                url = makeLocalImageAndUpload(name, croppedFrame)
                db.collection('embedding_name').document(name).update({'embedding_vectors': firestore.ArrayUnion([{'embedding_vector' : emb}]),
                'count' : len(db.collection('embedding_name').document(name).get().to_dict()['embedding_vectors'])+1,
                'image_paths' : firestore.ArrayUnion([{'image_path' : url}])})
            
                    

def makeCroppedFace(frame, resize = None):
    global arcface

    face_list = arcface.get(frame)
    if len(face_list) > 0:
        bbox = face_list[0]['bbox']
        if resize == None:
            face = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        else:
            try:
                face = cv2.resize(frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])], resize)
            except:
                # print(bbox)
                return 'error'
        return face
    else:
        return 'No face'

def preprocessFace(frame):
    croppedFrame = makeCroppedFace(frame, resize=(200, 200))
    # croppedFrame = makeCroppedFace(frame, resize=None)

    if croppedFrame == 'No face':
        return 'No face'
    elif croppedFrame == 'error':
        return 'error'

    else:
        encoded = cv2.imencode('.jpg', croppedFrame)[1]
        data = str(base64.b64encode(encoded))
        data = data[2:len(data)-1]
        return data

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=960,
    capture_height=540,
    display_width=960,
    display_height=540,
    framerate=20,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d !"
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "nvvidconv ! videobalance contrast=1 brightness=0.2 ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

async def transmit(websocket, path):
    global receiveName, isUploading, model, db
    try:
    
        if db.collection('camera_list').document(getmac.get_mac_address()).get().to_dict()['state'] == 'Occupied':
            await websocket.send('Occupied')
        else:
            await websocket.send('Connected')
            db.collection('camera_list').document(getmac.get_mac_address()).update({'state' : 'Occupied'})
            
            print("Client Connected")
            
            
            cam = cv2.VideoCapture('/dev/video1', cv2.CAP_V4L2)
            # cam = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

            while True:
                    ret, frame = cam.read()
                    if not ret:
                        print("fail to grab frame, try again")
                        break
                    
                    encoded = cv2.imencode('.jpg', frame)[1]
                    data = str(base64.b64encode(encoded))
                    data = data[2:len(data)-1]

                    if receiveName == True:
                        # threading.Thread(target = updateFirebase,args = (frame, name)).start()
                        data = preprocessFace(frame)
                        await websocket.send('crop')
                        while True:
                            if receiveName == False:
                                if isUploading:
                                    threading.Thread(target = updateFirebase,args = (frame, name), daemon = True).start()
                                    if not glob.glob('audio/name/' + name + '*'):
                                        threading.Thread(target = inference, args = (model, [name + '님'], 'ailab_joodw-neutral', 'audio/name/'), daemon = true).start()
                                    isUploading = False
                                break
                            await websocket.send(data)
                            await asyncio.sleep(0.0001)
                    
                    await websocket.send(data)
                    await asyncio.sleep(0.0001)
                    

    except:
        print("Client Disconnected ! - transmit")
        receiveName = False
        cam.release()
        db.collection('camera_list').document(getmac.get_mac_address()).update({'state' : 'Available'})
    


        
async def receive(websocket, path):
    global receiveName, name, isUploading
    try:
        while True:
            # 클라이언트로부터 메시지를 대기한다.
            data = await websocket.recv()
            if len(data) > 0:
                print("receive : " + data)
                print('dtype', type(data))
                if data == 'cancel':
                    receiveName = False
                elif data == 'ok':
                    isUploading = True
                    receiveName = False

                # elif data == 'Connecting':
                #     db.collection('camera_list').document(getmac.get_mac_address()).update({'state' : 'Occupied'})
                
                # 앱에서 이름 입력시
                else:
                    receiveName = True
                    name = data
    except:
        print("Client Disconnected ! - receive")
        receiveName = False
        #cv2.VideoCapture(0).release()
        db.collection('camera_list').document(getmac.get_mac_address()).update({'state' : 'Available'})

def exitHandler():
    global receiveName
    receiveName = False

def runServer():
    #db.collection('camera_list').document(getmac.get_mac_address()).update({'state' : 'Available'})
    atexit.register(close_camera)
    send_server = websockets.serve(transmit, port=8080)
    accept_server = websockets.serve(receive, port=8000, ping_interval = None)

    asyncio.get_event_loop().run_until_complete(asyncio.gather(send_server, accept_server))
    asyncio.get_event_loop().run_forever()

if __name__ == '__main__':

    db.collection('camera_list').document(getmac.get_mac_address()).update({'state' : 'Available'})
    runServer()
