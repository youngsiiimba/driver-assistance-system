from arduinoData import readSerial
import time



data = readSerial()
while True:
    if data.startswith("Location: INVAL"):
        print("INVALID")
    else:
        location = data.split('Date')[0]
        LatLong = location.replace("Location:", "")
        Lat = float(LatLong.split(",")[0])
        Long = float(LatLong.split(",")[0])
        print(f"Lat:{Lat} Long:{Long}")
        time.sleep(1)