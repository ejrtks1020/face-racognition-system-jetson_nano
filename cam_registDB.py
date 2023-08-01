import socket
import getmac
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# cred = credentials.Certificate('mykey.json')
# firebase_admin.initialize_app(cred, {
#     'databaseURL' : 'https://arface-79a8a-default-rtdb.firebaseio.com/'
# })

# db = firestore.client()
def extract_ip():
    st = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:       
        st.connect(('10.255.255.255', 1))
        IP = st.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        st.close()
    return IP

def close_camera():
    db = firestore.client() 
    macAddress = getmac.get_mac_address()
    db.collection('camera_list').document(macAddress).update({'state' : 'Unavailable'})

def regist_camera(db):    
    macAddress = getmac.get_mac_address()
    ipAddress = extract_ip()
    port = 8000
    camera_cnt =len(list(db.collection('camera_list').get()))
    data = {
        'ip' : ipAddress,
        'port' : port,
        'name' : 'camera' + str(camera_cnt),
        'state': 'Available',
    }
    print("MAC Address :", getmac.get_mac_address())
    
    print("IP Address(Internal) : ", ipAddress)

    print('Current # of camera : ',len(list(db.collection('camera_list').get())))

    if db.collection('camera_list').document(macAddress).get().exists:
        if ipAddress == db.collection('camera_list').document(macAddress).get().to_dict()['ip']:
            print('this camera exist in db')
        else:
            db.collection('camera_list').document(macAddress).update(
                {
                    'ip' : ipAddress
                }
            )
            print('Update Camera ip')
        #db.collection('camera_list').document(macAddress).update({'state' : 'Available'})
    else:
        db.collection('camera_list').document(macAddress).set(
            data
        )
        print('Regist New Camera')
    
    return db.collection('camera_list').document(macAddress).get().to_dict()['name'], getmac.get_mac_address()

# db = firestore.client()
# macAddress = getmac.get_mac_address()
# ipAddress = socket.gethostbyname(socket.gethostname())
# port = 8080
# print(db.collection('camera_list').document(macAddress).get().to_dict())

# if __name__ == '__main__':
#     # regist_camera(db)



