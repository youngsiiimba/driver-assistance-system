import urllib.request
import requests
import threading
import json
import time


# Define a function that will post on server every 15 Seconds
def post_data(state, Lat, Long):
    #threading.Timer(15,thingspeak_post).start()
    

    URl='https://api.thingspeak.com/update?api_key='
    KEY='OYFVTINBNHBNCOG7'
    HEADER='&field1={}&field2={}&field3={}'.format(state, Lat, Long)
    NEW_URL=URl+KEY+HEADER
    data=urllib.request.urlopen(NEW_URL)
    time.sleep(0.5)
#####################################################



#Function to collect data from ThingsSpeak  
def read_data():
    URL='https://api.thingspeak.com/channels/1833675/feeds.json?api_key='
    KEY='0VBO2FRT6OA9NTWK'
    HEADER='&results=1'
    NEW_URL=URL+KEY+HEADER

    get_data=requests.get(NEW_URL).json()
    channel_id=get_data['channel']['id']
    feeds=get_data['feeds']


    state=[]
    latitude=[]
    longitude=[]

    for x in feeds:
        state.append(x['field1'])
        latitude.append(x['field2'])
        longitude.append(x['field2'])
    
    print("Driver state: ", state)
    print(f"Car location: {} , {}", latitude, longitude)