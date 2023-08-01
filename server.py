
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
#from emb_data import db, arcface, bucket
from emb_data import q
import threading
from inference import get_model, inference
import glob
import logging

logging.basicConfig(level = logging.CRITICAL)

db = firestore.client()
bucket = storage.bucket()

#arcface = FaceAnalysis(name = 'buffalo_sc', providers = ['CPUExecutionProvider'])
# arcface = FaceAnalysis(name = 'buffalo_sc', providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider'])
#arcface.prepare(ctx_id=0, det_size=(640, 640))
receiveName = False
name = None
isUploading = False
isVideoCapturing = False
model = get_model()
frame_video = []
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

def updataFirebaseFromVideo():
    global frame_video
    frames = frame_video.copy()
    frame_video = []
    embs = []
    urls = []
    for frame in frames:
        
        image, face_list = frame[0], frame[1]
        croppedFrame = makeCroppedFace(image, face_list)
        if type(croppedFrame) != str:
            face = face_list[0]
            emb = list(face['embedding'].astype(float))
            url = makeLocalImageAndUpload(name, croppedFrame)
            embs.append(emb)
            urls.append(url)
    if embs:
        capture_time = time.strftime("%m-%d-%H-%M-%S", time.localtime())
        if not db.collection('embedding_name').document(name).get().exists:
        
            db.collection('embedding_name').document(name).set({
                'name': name,
                'embedding_vectors': [{f'{capture_time}_{i}' : emb} for i, emb in enumerate(embs)],
                'count': len(embs),
                'image_paths' : [{'image_path' : url} for url in urls]
                })
        else:
            db.collection('embedding_name').document(name).update({'embedding_vectors': firestore.ArrayUnion([{f'{capture_time}-{i}' : emb} for i, emb in enumerate(embs)]),
            'count' : len(db.collection('embedding_name').document(name).get().to_dict()['embedding_vectors'])+len(embs),
            'image_paths' : firestore.ArrayUnion([{'image_path' : url} for url in urls])})        

            
            


def updateFirebase(frame, face_list, name):
    global db
    # face_dict_list = arcface.get(frame)
    croppedFrame = makeCroppedFace(frame, face_list)

    if len(face_list) > 0:
        face = face_list[0]

        if face['det_score'] > 0.35:    
            emb = list(face['embedding'].astype(float))
            capture_time = time.strftime("%m-%d-%H-%M-%S", time.localtime())
            if not db.collection('embedding_name').document(name).get().exists:
                url = makeLocalImageAndUpload(name, croppedFrame)
                db.collection('embedding_name').document(name).set({
                    'name': name,
                    'embedding_vectors': [{f'{capture_time}' : emb}],
                    'count': 1,
                    'image_paths' : [{'image_path' : url}]
                    })
              
            else:
                url = makeLocalImageAndUpload(name, croppedFrame)
                db.collection('embedding_name').document(name).update({'embedding_vectors': firestore.ArrayUnion([{f'{capture_time}' : emb}]),
                'count' : len(db.collection('embedding_name').document(name).get().to_dict()['embedding_vectors'])+1,
                'image_paths' : firestore.ArrayUnion([{'image_path' : url}])})
            
                  


def makeCroppedFace(frame, face_list ,resize = None):
    # face_list = arcface.get(frame)
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

def preprocessFace(frame, face_list):
    croppedFrame = makeCroppedFace(frame, face_list, resize=(200, 200))
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

async def transmit(websocket, path):
    global receiveName, isUploading, model, db, frame_video, isVideoCapturing
    try:
    
        if db.collection('camera_list').document(getmac.get_mac_address()).get().to_dict()['state'] == 'Occupied':
            await websocket.send('Occupied')
        else:
            await websocket.send('Connected')
            db.collection('camera_list').document(getmac.get_mac_address()).update({'state' : 'Occupied'})
            
            print("Client Connected")
            
            
            # cam = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            # cam = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
            # cam = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)
            cnt = 0
            while True:
                    #ret, frame = cam.read()
                    #frame = cv2.resize(frame, (1024, 600))
                    
                    #if not ret:
                    #    print("fail to grab frame, try again")
                    #    break
                    if q.qsize() > 0:
                        data = q.get()                       
                        frame, face_dict_list = data[0], data[1]
                        #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                        encoded = cv2.imencode('.jpg', cv2.resize(frame, (300, 512)))[1]
                        data = str(base64.b64encode(encoded))
                        data = data[2:len(data)-1]

                        if receiveName == True:
                            if not isVideoCapturing:
                                data = preprocessFace(frame, face_dict_list)
                                # await websocket.send('crop')
                                while True:
                                    q.get()
                                    if receiveName == False:
                                        if isUploading:
                                            threading.Thread(target = updateFirebase,args = (frame, face_dict_list, name), daemon = True).start()
                                            if not glob.glob('audio/name/' + name + '*'):
                                                threading.Thread(target = inference, args = (model, [name + '님,'], 'ailab_joodw-neutral', 'audio/name/')).start()
                                            isUploading = False
                                        break
                                    await websocket.send(data)
                                    await asyncio.sleep(0.01)
                            if isVideoCapturing:
                                if cnt % 10 == 0:
                                    frame_video.append((frame, face_dict_list))
                                    # print(cnt)
                                    # threading.Thread(target = updateFirebase,args = (frame, face_dict_list , name)).start()
                        await websocket.send(data)
                        await asyncio.sleep(0.01)
                        cnt += 1
                    

    except Exception as e:
        print(e)
        print("Client Disconnected ! - transmit")
        receiveName = False
        isVideoCapturing = False
        frame_video = []
        db.collection('camera_list').document(getmac.get_mac_address()).update({'state' : 'Available'})
        while not q.empty():
            q.get()
        # cam.release()
        
    


        
async def receive(websocket, path):
    global receiveName, name, isUploading, isVideoCapturing, frame_video
    try:
        while True:
            # 클라이언트로부터 메시지를 대기한다.
            data = await websocket.recv()
            if len(data) > 0:
                print("receive : " + data)
                print('dtype', type(data))
                if data == 'cancel':
                    receiveName = False
                    isVideoCapturing = False
                    frame_video = []

                elif 'ok' in data:
                    if not 'video' in data:
                        isUploading = True
                    else:
                        threading.Thread(target = updataFirebaseFromVideo).start()
                        if not glob.glob('audio/name/' + name + '*'):                            
                            threading.Thread(target = inference, args = (model, [name + '님,'], 'ailab_joodw-neutral', 'audio/name/')).start()
                    receiveName = False
                    isVideoCapturing = False               
                # 앱에서 이름 입력시
                else:
                    if 'video' in data:
                        print(data)
                        isVideoCapturing = True
                        name = data.split('.')[0]
                    else:
                        name = data
                    receiveName = True
    except Exception as e:
        print(e)
        print("Client Disconnected ! - receive")
        receiveName = False
        isVideoCapturing = False
        frame_video = []
        #cv2.VideoCapture(0).release()
        db.collection('camera_list').document(getmac.get_mac_address()).update({'state' : 'Available'})

def exitHandler():
    global receiveName
    receiveName = False

def runServer():
    db.collection('camera_list').document(getmac.get_mac_address()).update({'state' : 'Available'})
    atexit.register(close_camera)
    send_server = websockets.serve(transmit, port=8080)
    accept_server = websockets.serve(receive, port=8000, ping_interval = None)

    asyncio.get_event_loop().run_until_complete(asyncio.gather(send_server, accept_server))
    asyncio.get_event_loop().run_forever()

if __name__ == '__main__':

    db.collection('camera_list').document(getmac.get_mac_address()).update({'state' : 'Available'})
    runServer()
