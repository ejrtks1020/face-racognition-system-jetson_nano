import insightface
import time
import cv2

img = cv2.imread('images2.jpeg')
print(img.shape)
img_resize = cv2.resize(img, (640, 640))
start = time.time()
detector = insightface.model_zoo.get_model('det_10g.onnx', providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider'])
detector.prepare(ctx_id = 0)


recognition = insightface.model_zoo.get_model('w600k_r50.onnx', providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider'])
recognition.prepare(ctx_id = 0)
print(type(recognition))
print('model load time: ' , time.time() - start)

start = time.time()
result = detector.detect(img, input_size = (640, 640))

faces = []
kps = []
for i, k in zip(result[0], result[1]):
    bbox = list(map(int, i[:4]))
    score = i[4]
    if score > 0.5:
        face = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        while True:
          cv2.imshow("IMG", face)
          k = cv2.waitKey(1)
          if k%256==27: # ESC
            print('Esc pressed, closing...')
            
            break
        faces.append(face)
        kps.append(k)
# cv2.imshow(faces[0])
# cv2.imshow(faces[1])
print(result)
print(faces)
print('det model inference time: ' , time.time() - start)
results = []
for f, k in zip(faces, kps):
    key = {'kps' : k}
    results.append(recognition.get(img, key))

print(results)
