import cv2
#from arduinoData import readSerial
import time

IMG_SIZE = 145
def prepare(filepath, face_cas="/miscellaneous/haarcascade_frontalface_default.xml"):
    #img_array = cv2.imread(filepath, cv2.IMREAD_COLOR)
    img_array = filepath
    img_array = img_array / 255
    resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return resized_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)


# def GPSdata():
#     data = readSerial()
#     if data.startswith("Location: INVAL"):
#         return 0.0, 0.0
#     else:
#         location = data.split('Date')[0]
#         LatLong = location.replace("Location:", "")
#         Lat = float(LatLong.split(",")[0])
#         Long = float(LatLong.split(",")[0])
#         return Lat, Long
# while True:
#     Lat , Long = GPSdata()
#     print(f"{Lat} and {Long}")
#     time.sleep(1)




