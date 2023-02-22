from keras.models import load_model
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2

#WORKING PROJECT
import numpy as np
import argparse
import pickle
import cv2
import os
import time 
from keras.models import load_model
from collections import deque
import telepot

import datetime
import cv2 
import time
import datetime
import telepot
import serial

import time
print("1 Violence Detection/2 Metal Detection")
a=input("enter data")
if a=='1':
    


    token = '5418643105:AAGh8LmQed84DzW7mdfQvHonQkiJ1C0imjQ' # telegram token
    receiver_id = 5105004126
    
    def print_results(video, limit=None):
            #fig=plt.figure(figsize=(16, 30))
            if not os.path.exists('output'):
                os.mkdir('output')
    
            print("Loading model ...")
            model = load_model(r'C:\Users\praju\OneDrive\Desktop\Violence\modelnew.h5')
            Q = deque(maxlen=128)
            vs = cv2.VideoCapture(video)
            writer = None
            frame_count = 0
            (W, H) = (None, None)
            count = 0     
            while True:
                # read the next frame from the file
                (grabbed, frame) = vs.read()
    
                # if the frame was not grabbed, then we have reached the end
                # of the stream
                if not grabbed:
                    break
                
                # if the frame dimensions are empty, grab them
                if W is None or H is None:
                    (H, W) = frame.shape[:2]
    
                # clone the output frame, then convert it from BGR to RGB
                # ordering, resize the frame to a fixed 128x128, and then
                # perform mean subtraction
    
                
                output = frame.copy()
               
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (128, 128)).astype("float32")
                frame = frame.reshape(128, 128, 3) / 255
    
                # make predictions on the frame and then update the predictions
                # queue
                preds = model.predict(np.expand_dims(frame, axis=0))[0]
    #             print("preds",preds)
                Q.append(preds)
    
                # perform prediction averaging over the current history of
                # previous predictions
                results = np.array(Q).mean(axis=0)
                i = (preds > 0.50)[0]
                label = i
    
                text_color = (0, 255, 0) # default : green
    
                if label: # Violence prob
                    text_color = (0, 0, 255) # red
    
                else:
                    text_color = (0, 255, 0)
    
                text = "Violence: {}".format(label)
                FONT = cv2.FONT_HERSHEY_SIMPLEX 
    
                cv2.putText(output, text, (35, 50), FONT,1.25, text_color, 3) 
            
                   
                if writer is None:
                    
                    if(text=="Violence: True"):
                        print("Violence Detected")
                        date_string = datetime.datetime.now().strftime("%Y-%m-%d  %I.%M.%S%p   %A")
                        time=date_string[10:17]
                        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                        writer = cv2.VideoWriter(time+'.avi', fourcc, 30,(W, H), True)
                        con,frames = vs.read()   
                        if con:
                            name = str(frame_count)+'.jpg'
                            print('Capturing --- '+name)
                            for j in range(0,1):
                                   
                                cv2.imwrite('Frame'+str(j)+'.jpg',frames)
                               
                           
                     
                               # Extracting images and saving with name 
                               #cv2.imwrite(name, frames) 
                                frame_count = frame_count + 1
                                bot = telepot.Bot(token)
    
                                bot.sendMessage(receiver_id, 'Violence Detected') 
                                bot.sendMessage(receiver_id, 'Camera1')
                                bot.sendPhoto(receiver_id, photo=open('Frame0.jpg', 'rb'))
                                writer.write(output)
                    elif(text=="Violence == Fasle"):
                        print("Not Detected")
                        writer.write(output)
               
                   
                    
                   
                
                        
                
                        
                
                   
               
                    
                        
                   
    
                            
                
               
    
                # check if the video writer is None
                
    
                # write the output frame to disk
               
    
                # show the output image
                cv2.imshow("video",output)
                key = cv2.waitKey(1) & 0xFF
    
                # if the `q` key was pressed, break from the loop
                if key == ord("q"):
                    break
            # release the file pointersq
            print("[INFO] cleaning up...")
            writer.release()
            vs.release()
    
    
    
    V_path = r"C:\Users\praju\OneDrive\Desktop\Violence\vv.mp4" 
    NV_path =r"C:\Users\praju\OneDrive\Desktop\Violence\nonvv.mp4"
    
    print_results(V_path)

elif(a=='2'):
  
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    serialcom = serial.Serial('COM3', 9600)

    serialcom.timeout = 1


    token = '5418643105:AAGh8LmQed84DzW7mdfQvHonQkiJ1C0imjQ' # telegram token
    receiver_id = 5105004126
      
    #face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    #width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap = cv2.VideoCapture(0)
    i=0
    date_string = datetime.datetime.now().strftime("%Y-%m-%d  %I.%M.%S%p   %A")
    time=date_string[10:17]
    out= cv2.VideoWriter(time+'.avi', cv2.VideoWriter_fourcc(*'DIVX'), 20, (640,480))
    #width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #writer= cv2.VideoWriter('12.avi', cv2.VideoWriter_fourcc(*'DIVX'), 20, (width,height))
    while True:
        #time.sleep(0.5)
        a=serialcom.readline(35).decode('utf-8').rstrip()
        #print(a)
        if(a=="1"):
            serialcom.close()  
            z=1
            while(True):
                ret, frame = cap.read() 
              
                # Converts to HSV color space, OCV reads colors as BGR
                # frame is converted to hsv
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                frame = cv2.resize(frame, (640, 480))
                boxes, weights = hog.detectMultiScale(frame, winStride=(8,8) )

                boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

                for (xA, yA, xB, yB) in boxes:
                    # display the detected boxes in the colour picture
                    cv2.rectangle(frame, (xA, yA), (xB, yB),
                                      (0, 255, 0), 2)
                font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
                dt = str(datetime.datetime.now())
                timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]
                cv2.putText(frame, dt,
                                    (10, 100),
                                    font, 1,
                                    (0, 255, 0),
                                    2, cv2.LINE_8)
        
                #cv2.imshow("Frame",frame)
                cv2.imwrite('Frame'+str(i)+'.jpg', frame)
                
                
                out.write(frame) 
                  
                # The original input frame is shown in the window 
                cv2.imshow('Original', frame)
                if(z==1):
                    
                    bot = telepot.Bot(token)
                
                    bot.sendMessage(receiver_id, 'Metal Detected') 
                    #bot.sendMessage(receiver_id, 'Adipidi at Dhaaravi')
                    bot.sendPhoto(receiver_id, photo=open('Frame0.jpg', 'rb'))
                    z=2
              
                # The window showing the operated video stream 
              
                  
                # Wait for 'a' key to stop the program 
                if cv2.waitKey(1) & 0xFF == ord('a'):
                    break
              
            # Close the window / Release webcam
            cap.release()
              
            # After we release our webcam, we also release the output
            out.release() 
              
            # De-allocate any associated memory usage 
            cv2.destroyAllWindows()
            
                
            # reads frames from a camera 
            # ret checks return at each frame
            