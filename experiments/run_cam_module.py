from cam_main_no_multi import runFaceRecognition
from server import runServer
import multiprocessing
if __name__ == '__main__':

  multiprocessing.Process(target=runFaceRecognition).start()
  multiprocessing.Process(target=runServer).start()
