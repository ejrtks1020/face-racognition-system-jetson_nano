from fileinput import filename
import sklearn
import tensorflow as tf
import torch
# from torchvision import datasets
from torch.utils.data import DataLoader
from insightface.app import FaceAnalysis
import os
import glob
import numpy as np
import cv2
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import storage
from google.cloud.firestore_v1 import Increment
import random
import multiprocessing as mp
q = mp.Manager().Queue()
cred = credentials.Certificate('xinapse-key.json')
firebase_admin.initialize_app(cred, {
    'storageBucket' : 'xinapse-face-recognition.appspot.com'
})

db = firestore.client()

bucket = storage.bucket()

#arcface = FaceAnalysis(name = 'buffalo_l', providers=['TensorrtExecutionProvider'], allowed_modules =['detection','recognition'])
arcface = FaceAnalysis(name = 'buffalo_sc', providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider'])
#arcface = FaceAnalysis(name = 'buffalo_sc', providers = ['CPUExecutionProvider'])
# arcface = FaceAnalysis(providers=['CoreMLExecutionProvider'])
# arcface = FaceAnalysis()

arcface.prepare(ctx_id=0, det_thresh = 0.45, det_size=(640, 640))

def collate_fn(x):
    return x[0]

def fileUpload(fname, name):
  hash_name = "%032x" % random.getrandbits(128)
  blob = bucket.blob('registered_person/' + name + '/' + hash_name)
  blob.upload_from_filename(fname)
  blob.make_public()
  return blob.public_url


def db_list():
  img_dir = 'static/photos'
  for (root, dirs, files) in os.walk(img_dir):
    if len(files) > 0:
        for file_name in files:
          file_path = os.path.join(root, file_name)
          img_array = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
          face_dict_list = arcface.get(img_array)

          if len(face_dict_list) > 0:
            face = face_dict_list[0]

            if face['det_score'] > 0.5:
              emb = list(face['embedding'].astype(float))
              name = root.split('/')[-1]
              #db.collection('embedding_name2').add({'embedding_vector': emb, 'name' : name})
              # 새로운 name을 등록하는 경우
              if not db.collection('embedding_name').document(name).get().exists:
                url = fileUpload(file_path, name)
                db.collection('embedding_name').document(name).set({
                  'name': name,
                  'embedding_vectors': [{'embedding_vector' : emb}],
                  'count': 1,
                  'image_paths' : [{'image_path' : url}]
                  
                  })
                
              # 기존에 등록된 name을 추가로 등록하는 경우
              else:
                before_count = len(db.collection('embedding_name').document(name).get().to_dict()['embedding_vectors'])
                db.collection('embedding_name').document(name).update({'embedding_vectors': firestore.ArrayUnion([{'embedding_vector' : emb}])})
                after_count = len(db.collection('embedding_name').document(name).get().to_dict()['embedding_vectors'])
                # 만약 같은 이미지가 등록되어 있으면 업데이트를 하지않는다.
                if before_count != after_count:
                  url = fileUpload(file_path, name)
                  db.collection('embedding_name').document(name).update({
                    'count' : after_count,
                    'image_paths' : firestore.ArrayUnion([{'image_path' : url}])})
                  

              
              

              
              #db.collection('embedding_name').document(name).collection('embedding').add({'embedding_vector': emb, 'name' : name})



  # 폴더로부터 데이터를 읽는다.
  # dataset = datasets.ImageFolder('static/photos') # 이미지 경로
  # idx_to_class = {i:c for c,i in dataset.class_to_idx.items()} # 각 클래스의 인덱스를 이름으로 매핑하는 딕셔너리
  # loader = DataLoader(dataset, collate_fn=collate_fn)

  # for img, idx in loader:
  #     img_array = np.array(img)
  #     face_dict_list = app.get(img_array)
  #     if len(face_dict_list) > 0:
  #       face = face_dict_list[0]
  #       if face['det_score'] > 0.5:
  #         emb = list(face['embedding'])
  #         embedding_list.append(emb) 
  #         name_list.append(idx_to_class[idx])       
  # save data
  #data = [embedding_list, name_list] 
  #torch.save(data, 'data.pt') # saving data.pt file
if __name__ == '__main__':
  db_list()
  
