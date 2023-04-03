import serial
import serial.tools.list_ports

ports = serial.tools.list_ports.comports()
serialInst = serial.Serial()

## List ports
portList = []
for port in ports:
    portList.append(str(port))
    #print("------------------------------------")
    #print("COM ports detected:")
    #print(str(port))
    #print("------------------------------------ \n")

#val = input("Pick a COM port(e.g 'COM4'):  ")
val = "COM4"
for x in range(0, len(portList)):
    if portList[x].startswith(str(val)):
        portVar = str(val)
        #print(portVar)
        #print(portList[x])

serialInst.baudrate = 115200
serialInst.port = portVar
serialInst.open()

def readSerial():
    while True:
        if serialInst.in_waiting:
            packet = serialInst.readline()  
            #print(packet.decode('utf')) 
            return packet.decode('utf') 
readSerial()