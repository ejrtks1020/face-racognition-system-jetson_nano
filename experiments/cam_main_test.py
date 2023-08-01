import cv2
import time
model_load = time.time()
from emb_data import arcface
print('model load time: ', time.time() - model_load)


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d !"
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
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

#cam = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
def runFaceRecognition():

  cam = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
  # width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
  # height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
  # print('original size : %d, %d' % (width, height))

  # cam.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
  # cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
  # width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
  # height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
  # print('changed size : %d, %d' % (width, height))

  while True:

    ret, frame = cam.read()
    start = time.time()
    arcface.get(frame)
    print('get emb vector: ', time.time() - start)
    # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    if not ret:
        print("fail to grab frame, try again")
        break
   
    cv2.imshow("IMG", frame)


    
            
    k = cv2.waitKey(1)
    if k%256==27: # ESC
        print('Esc pressed, closing...')
        break

  cam.release()
  cv2.destroyAllWindows()





if __name__ == '__main__':
  runFaceRecognition()



