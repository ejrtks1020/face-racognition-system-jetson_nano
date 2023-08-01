import re
from cv2 import threshold
import tensorflow
import torch
import random
import cv2
import os
import time
from PIL import ImageFont, ImageDraw, Image
from play_tts import playTTS
from emb_data import arcface
import numpy as np
from numpy import dot, rec
from numpy.linalg import norm
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore, storage
from firebase_admin.firestore import SERVER_TIMESTAMP
from emb_data import db_list, db, bucket
import websockets
import asyncio
import struct, pickle, base64
import threading
from cam_registDB import regist_camera, close_camera
import atexit
import timeit
import warnings
from annoy import AnnoyIndex
warnings.filterwarnings(action='ignore')

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def cosine_similarity(x, y):
  return dot(x, y)/(norm(x)*norm(y))
    
def on_snapshot(col_snapshot, changes, read_time):
    global embedding_list, name_list, time_person, callback_done
    for change in changes:
        if change.type.name == 'ADDED':
            
            for emb in change.document.to_dict()['embedding_vectors']:
                embedding_list.append(emb['embedding_vector'])
                name_list.append(change.document.to_dict()['name'])
                time_person[change.document.to_dict()['name']] = 0
        
        elif change.type.name == 'MODIFIED':
            #print(change.document.to_dict())
            print(name_list.count(change.document.to_dict()['name']), change.document.to_dict()['count'])
            if name_list.count(change.document.to_dict()['name']) < change.document.to_dict()['count']:
              embedding_list.append(change.document.to_dict()['embedding_vectors'][-1]['embedding_vector'])
              name_list.append(change.document.to_dict()['name'])
            else:
              # print(change.document.to_dict())
              # print(change.document.to_dict()['embedding_vectors'][-1]['embedding_vector'])

              # print(len(name_list))
              # print(len(embedding_list))
              
              # start = time.time()
              # print(len(embedding_list))
              # print(len(name_list))

              # name_index = [i for i in range(len(name_list)) if name_list[i] == change.document.to_dict()['name']]
              # name_list = [name for i, name in enumerate(name_list) if i not in name_index]
              # embedding_list = [emb for i, emb in enumerate(embedding_list) if i not in name_index]

              # db_embs = [d['embedding_vector'] for d in change.document.to_dict()['embedding_vectors']]
              # db_names = [change.document.to_dict()['name']] * len(db_embs)

              # embedding_list.extend(db_embs)
              # name_list.extend(db_names)

              # print(time.time() - start)
              # print(len(embedding_list))
              # print(len(name_list))

              # start = time.time()
              # print(len(embedding_list))
              # print(len(name_list))

              name_index = [i for i in range(len(name_list)) if name_list[i] == change.document.to_dict()['name']]
              

              db_embs = [d['embedding_vector'] for d in change.document.to_dict()['embedding_vectors']]
              # db_names = [change.document.to_dict()['name']] * len(db_embs)
              # print(name_index)
              # #print(db_embs)
              # print(len(db_embs))

              index_delete = []

              for i in name_index:
                if embedding_list[i] not in db_embs:
                  index_delete.append(i)
              
              embedding_list = [emb for i, emb in enumerate(embedding_list) if i not in index_delete]
              name_list = [name for i, name in enumerate(name_list) if i not in index_delete]

              # print(time.time() - start)
              # print(len(embedding_list))
              # print(len(name_list))

              

        elif change.type.name == 'REMOVED':
          if name_list.count(change.document.id) > 0:
            del embedding_list[name_list.index(change.document.id)]
            del name_list[name_list.index(change.document.id)]
              

    callback_done.set()

def on_snapshot_cam(col_snapshot, changes, read_time):
    global cam_mac
    for change in changes:        
        if change.type.name == 'MODIFIED' and change.document.id == cam_mac:
            cam_name = change.document.to_dict()['name']
        
    callback_done_cam.set()

def fileUpload(type, fname, hash):
  blob = bucket.blob(os.path.join(type, hash))
  blob.upload_from_filename(fname)
  blob.make_public()
  return blob.public_url


def personLog(db, cam_name, logType, hash, frame, logKnown = None, logUnknownNum = None):
  if not os.path.exists(logType):
      os.mkdir(logType)
  file_path = os.path.join(logType, hash+ '.jpg')
  cv2.imwrite(file_path, frame)
  url = fileUpload(logType, file_path, hash)
  os.remove(file_path)
  if logType == 'Registered_person_log':
    db.collection(logType).document(hash).set({
    'captured_time' : SERVER_TIMESTAMP,
    'name_list' : logKnown,
    'camera' : cam_name,
    'url' : url})
  elif logType == 'Unregistered_person_log':
    db.collection(logType).document(hash).set({
    'captured_time' : SERVER_TIMESTAMP,
    'Unknowns' : logUnknownNum,
    'camera' : cam_name,
    'url' : url})



def uploadLogdata(db, cam_name, log_known, log_unknown_num, hash, frame):
  frame_resized = cv2.resize(frame, (480, 270))

  if len(log_known) > 0:
    personLog(db, cam_name, 'Registered_person_log', hash, frame_resized, logKnown=log_known)
    if log_unknown_num > 0:
      personLog(db, cam_name, 'Unregistered_person_log', hash, frame_resized, logUnknownNum=log_unknown_num)
      
  elif len(log_known) == 0 and log_unknown_num > 0:
    personLog(db, cam_name, 'Unregistered_person_log', hash, frame_resized, logUnknownNum=log_unknown_num)

def uploadUnknownVector(db, emb):
  if not db.collection('embedding_name').document('unknown').get().exists:
                db.collection('embedding_name').document('unknown').set({
                  'embedding_vectors': [{'embedding_vector' : emb}],
                  'count': 1,                  
                  })
  else:
    count = len(db.collection('embedding_name').document('unknown').get().to_dict()['embedding_vectors'])
    db.collection('embedding_name').document('unknown').update({'embedding_vectors': firestore.ArrayUnion([{'embedding_vector' : emb}]),
                                                          'count': count+1})

def playTTSandUpdateTimeTerm(personTimeDict, recogPersons, known = True):
    global isSpeaking
    isSpeaking = True
    playTTS(recogPersons, known)
    for person in recogPersons:
      personTimeDict[person] = time.time()
    isSpeaking = False

callback_done = threading.Event()    
callback_done_cam = threading.Event()
embedding_list, name_list, unknown_emb_list = [], [], []
time_person = {}
time_unknown_person = {}
isSpeaking = False
# db에 카메라를 등록하고 등록된 카메라 명을 반환
cam_name, cam_mac = regist_camera(db)
def accuracyColor(accuracy):
  if accuracy < 0.5:
    color = (0, 0, 255) # 빨간색
  elif accuracy < 0.6:
    color = (0, 69, 255) # 오렌지 레드
  elif accuracy < 0.7:
    color = (71, 99, 255) # 토마토
  elif accuracy < 0.8:
    color = (0, 215, 255) # 금
  elif accuracy < 0.9:
    color = (0, 255, 127) # 연두빛
  elif accuracy <= 1:
    color = (0, 255, 0) # 초록색
  
  return color

def makeAnnoyIndex(embedding_list, filename):
    annoy_index = AnnoyIndex(512, 'angular')
    for idx, emb_db in enumerate(embedding_list):
      annoy_index.add_item(idx, emb_db)
          
    annoy_index.build(10)
    annoy_index.save(filename)
    new_annoy_index = AnnoyIndex(512, 'angular')
    new_annoy_index.load(filename)

    return new_annoy_index

def drawBboxAndText(frame, box, text, min_dist):
    
    bbox_width = int(box[2]) - int(box[0])
    bbox_height = int(box[3]) - int(box[1])
    
    font = ImageFont.truetype("font/NanumGothicExtraBold.ttf", int(bbox_width * 0.09))

    resized_bbox_pt1_x = int(int(box[0]) - bbox_width * 0.1)
    resized_bbox_pt1_y = int(int(box[1]) - bbox_height * 0.2)
    resized_bbox_pt2_x = int(int(box[2]) + bbox_width * 0.1)
    resized_bbox_pt2_y = int(box[3])

    resized_bbox_width = resized_bbox_pt2_x - resized_bbox_pt1_x
    resized_bbox_height = resized_bbox_pt2_y - resized_bbox_pt1_y

    # frame decorate 
    frame = cv2.rectangle(frame, (resized_bbox_pt1_x, int(resized_bbox_pt1_y - (resized_bbox_pt2_y - resized_bbox_pt1_y) * 0.08)) , (resized_bbox_pt2_x ,resized_bbox_pt1_y ), (0,0,255), -1)
    frame = cv2.rectangle(frame, (resized_bbox_pt1_x,resized_bbox_pt1_y) , (resized_bbox_pt2_x ,resized_bbox_pt2_y), (0,0,255), 2)
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.text((int(resized_bbox_pt1_x + resized_bbox_width * 0.05), int(resized_bbox_pt1_y - resized_bbox_height * 0.08)), text, font = font, fill = (0, 255, 0))

    frame = np.array(img)
          #frame = cv2.putText(origin_frame, text, (int(box[0]),int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2, cv2.LINE_AA)

          # print(frame.shape)
          # box[2], box[3]은 바운딩박스의 우하단 x, y 좌표
    if text != '미등록':
            # 좌 하단
      accuracy_bar_pt1_x = int(resized_bbox_pt2_x + resized_bbox_width * 0.02)
      accuracy_bar_pt1_y = int(resized_bbox_pt2_y)
            # 우 상단
      accuracy_bar_pt2_x = int(accuracy_bar_pt1_x + (resized_bbox_pt2_x - resized_bbox_pt1_x) * 0.05)
      accuracy_bar_pt2_y = accuracy_bar_pt1_y - int((resized_bbox_pt2_y - resized_bbox_pt1_y) * round(min_dist, 4))
            
      color = accuracyColor(round(min_dist, 4))
      frame = cv2.rectangle(frame, (accuracy_bar_pt1_x, accuracy_bar_pt1_y) , (accuracy_bar_pt2_x, accuracy_bar_pt2_y), color, -1)
      frame = cv2.putText(frame, str(int(min_dist * 100)) + '%', (accuracy_bar_pt1_x,accuracy_bar_pt2_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame

#cam = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
def runFaceRecognition():
  global embedding_list, name_list, unknown_emb_list
  global time_person, time_unknown_person, isSpeaking

  atexit.register(close_camera)
  # db = firestore.client()
  # bucket = storage.bucket()

  # db에 카메라를 등록하고 등록된 카메라 명을 반환
  # cam_name = regist_camera(db)

  isDelaying = False
  greet_delay_time = 0

  
  col_query = db.collection('embedding_name')
  col_query_cam = db.collection('cameral_list')
  #db_list()
  

  
  # 데이터베이스를 Listen하는 쿼리스냅샷
  query_watch = col_query.on_snapshot(on_snapshot)
  query_watch_cam = col_query_cam.on_snapshot(on_snapshot_cam)
  cam = cv2.VideoCapture('/dev/video1', cv2.CAP_V4L2)
  cv2.moveWindow("IMG", 0, -30)
  #print('before cam')
  #cam = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
  #pipeline = " ! ".join(["v4l2src device=/dev/video1",
  #                     "video/x-raw, width=640, height=480, framerate=30/1",
  #                     "videoconvert",
  #                     "video/x-raw, format=(string)BGR",
  #                     "appsink"
  #                     ])
  # cam = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
  # print('after cam')
  # width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
  # height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
  # print('original size : %d, %d' % (width, height))

  # cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
  # cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
  # width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
  # height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
  # print('changed size : %d, %d' % (width, height))


  time_start = time.time()
  while True:

    ret, frame = cam.read()
    frame = cv2.resize(frame, (1024, 600))
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # FPS check start
    start_t = timeit.default_timer()
    if not ret:
        print("fail to grab frame, try again")
        break
    time_end = time.time()
    face_dict_list = arcface.get(frame)
    log_known = []
    log_unknown_num = 0
    unknown_list = []
    dist_list = []
    if len(face_dict_list) > 0:
      for face in face_dict_list:
        if face['det_score'] > 0.5:
          
          emb = face['embedding']

          annoy_index = makeAnnoyIndex(embedding_list, 'face_embs.annoy')
          idx_list = annoy_index.get_nns_by_vector(emb, len(embedding_list), include_distances=True)
          # print(idx_list)
          
      

          # for idx, emb_db in enumerate(embedding_list):
          #   dist = 1 - cosine_similarity(np.array(emb), np.array(emb_db))
          #   dist_list.append(dist)
          # if len(dist_list) > 0:
          #   print('my cos : ' , sorted(dist_list))
            # min_dist = max(dist_list)
            # min_dist_idx = dist_list.index(min_dist)
            # name = name_list[min_dist_idx]          
            
          box = face['bbox']
          text = '미등록'
          if len(idx_list) > 0:
            # print('annoy cos : ', idx_list[1])

            min_dist = idx_list[1][0]
            min_dist_idx = idx_list[0][0]
            name = name_list[min_dist_idx]
            #print(min_dist)

            new_min_dist = 1 - (1 - (min_dist**2) / 2)
            
            accuracy = max(round((2 - new_min_dist) / 2, 2), 0.85)
          

            if new_min_dist < 0.45:
              #text = name+' '+str(round(new_min_dist, 4))
              text = name
              log_known.append(name)
          
          unknownDistList = []
          #해당 face가 등록이 안된 face일때
          if text == '미등록':
            log_unknown_num += 1
            # 기존 Unknown 벡터와 비교해서 새로운 Unknown인지 체크
            # for UnknownEmb in unknown_emb_list:
            #   dist = cosine_similarity(np.array(emb), np.array(UnknownEmb))
            #   unknownDistList.append(dist)

            annoy_index = makeAnnoyIndex(unknown_emb_list, 'unknown_face_embs.annoy')
            idx_list = annoy_index.get_nns_by_vector(emb, len(unknown_emb_list), include_distances=True)
            

            # 해당 카메라에서 첫 Unknown일때
            if len(idx_list[0]) == 0:
              unknown_emb_list.append(emb)
              time_unknown_person['Unknown0'] = 0
              #nametts = 'Unknown0'
              unknown_list.append('Unknown0')
            else:
              min_dist = idx_list[1][0]
              new_min_dist = 1 - (1 - (min_dist**2) / 2)
              min_dist_idx = idx_list[0][0] 
              #accuracy = round((2 - min_dist) / 2, 2)
              if new_min_dist > 0.45: # 새로운 Unknown이면 db에 등록
                
                time_unknown_person['Unknown'+str(len(unknown_emb_list))] = 0
                unknown_list.append('Unknown'+str(len(unknown_emb_list)))
                #nametts = 'Unknown'+str(len(unknown_emb_list))
                unknown_emb_list.append(emb)
                
                
              else: # 기존에 기록된 Unknown일때
                # minUnknownDist = max(unknownDistList)
                minUnknownDistIndex = min_dist_idx
                #nametts = 'Unknown'+str(minUnknownDistIndex)
                unknown_list.append('Unknown'+str(minUnknownDistIndex))
            #timeDict = time_unknown_person
          # else:
          #   nametts = name
          #   timeDict = time_person
            
            
          # if (time.time() - timeDict[nametts]) > 4 and isSpeaking == False:
          #   threading.Thread(target = playTTSandUpdateTimeTerm, args = (timeDict, nametts), daemon = True).start()




          frame = drawBboxAndText(frame, box, text, accuracy)


    if isSpeaking == False:
      
      if len(log_known) > 0:
        validpeople = [name for name in log_known if (time.time() - time_person[name] > 3)]
        timeDict = time_person
        known = True
        
      else:
        validpeople = [unknown for unknown in unknown_list if (time.time() - time_unknown_person[unknown] > 3)]
        timeDict = time_unknown_person
        known = False
      
      if len(validpeople) > 0:
        if isDelaying == False:
          greet_delay_time = time.time()
          isDelaying = True
          
        elif isDelaying == True and time.time() - greet_delay_time > 1:
          threading.Thread(target = playTTSandUpdateTimeTerm, args = (timeDict, validpeople, known), daemon = True).start()
          isDelaying = False

      else:
        if isDelaying == True and time.time() - greet_delay_time > 1:
          isDelaying = False



      
    # 전체 안면인식 알고리즘 종료
    # FPS check end
    terminate_t = timeit.default_timer()
    fps = str(int(1./(terminate_t -  start_t)))
    frame = cv2.putText(frame, fps, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2, cv2.LINE_AA)
          
    if time_end - time_start >= 3.0:    
      #captured_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
      hash = "%032x" % random.getrandbits(128)
      # threading.Thread(target = uploadLogdata, args=(db, cam_name, log_known, log_unknown_num, hash, frame)).start()
      # uploadLogdata(log_known=log_known, log_unknown_num=log_unknown_num, hash=hash)

      time_start = time_end
    
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #frame = cv2.rotate(frame, cv2.ROTATE_180)
    cv2.imshow("IMG", frame)


    
            
    k = cv2.waitKey(1)
    if k%256==27: # ESC
        print('Esc pressed, closing...')
        query_watch.unsubscribe()
        query_watch_cam.unsubscribe()
        break

  cam.release()
  cv2.destroyAllWindows()





if __name__ == '__main__':
  runFaceRecognition()



