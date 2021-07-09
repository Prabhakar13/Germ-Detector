from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import tkinter
import time
import random
import math

# width of the animation window
animation_window_width=900
# height of the animation window
animation_window_height=675

k_att=10
k_rep=100000
k_att_wo_m=10
k_rep_wo_m=10000
k_att_w_m=100
k_rep_w_m=200
#maximum velocity
vmax=10

# The main window of the animation
def create_animation_window():
  window = tkinter.Tk()
  window.title("Germ_Simulation")
  window.geometry(f'{animation_window_width}x{animation_window_height}')
  return window
 
# Create a canvas for animation and add it to main window
def create_animation_canvas(window):
  canvas = tkinter.Canvas(window)
  canvas.configure(bg="white")
  canvas.pack(fill="both", expand=True)
  return canvas


#function to create and returns all objects in the canvas
def create_objects(window,canvas):
    balls=[]
    for i in range(20):
        ball_pos_x=random.randint(20,800)
        ball_pos_y=random.randint(20,600)
        
        t_ball=canvas.create_oval(ball_pos_x,ball_pos_y,ball_pos_x+40,ball_pos_y+40,
                fill="green", outline="black", width=1)
        balls.append(t_ball)
    dest_pos_x=450
    dest_pos_y=300
    dest=canvas.create_oval(dest_pos_x,dest_pos_y,dest_pos_x+40,dest_pos_y+40,
        fill="red", outline="black", width=1)
        
    return balls,dest
def euclid_distance(x1,y1,x2,y2):
    return math.sqrt(pow(x1-x2,2)+pow(y1-y2,2))

def find_same_force_wo_mask(window,canvas,ball,dest):
    
    ball_pos_x,ball_pos_y,t1,t2 = canvas.coords(ball)
    ball_pos_x=(int)(ball_pos_x+t1)/2
    ball_pos_y=(int)(ball_pos_y+t2)/2
    
    dest_pos_x,dest_pos_y,t1,t2 = canvas.coords(dest)
    dest_pos_x=(int)(dest_pos_x+t1)/2
    dest_pos_y=(int)(dest_pos_y+t2)/2

    dist_dest_ball_x=dest_pos_x-ball_pos_x
    dist_dest_ball_y=dest_pos_y-ball_pos_y
    
    net_force_ikap=0
    net_force_jkap=0

    euclid_dist=euclid_distance(ball_pos_x,ball_pos_y,dest_pos_x,dest_pos_y)
    if(euclid_dist>100):
        net_force_ikap=(k_att_wo_m*dist_dest_ball_x)
        net_force_jkap=(k_att_wo_m*dist_dest_ball_y)
    else:
        net_force_ikap=-(k_rep_wo_m*dist_dest_ball_x)
        net_force_jkap=-(k_rep_wo_m*dist_dest_ball_y)

    net_force_ikap=net_force_ikap/max(1,abs(euclid_dist*(euclid_dist-40)))
    net_force_jkap=net_force_jkap/max(1,abs(euclid_dist*(euclid_dist-40)))

    return net_force_ikap,net_force_jkap


def find_same_force_w_mask(window,canvas,ball,dest):
    
    ball_pos_x,ball_pos_y,t1,t2 = canvas.coords(ball)
    ball_pos_x=(int)(ball_pos_x+t1)/2
    ball_pos_y=(int)(ball_pos_y+t2)/2
    
    dest_pos_x,dest_pos_y,t1,t2 = canvas.coords(dest)
    dest_pos_x=(int)(dest_pos_x+t1)/2
    dest_pos_y=(int)(dest_pos_y+t2)/2

    dist_dest_ball_x=dest_pos_x-ball_pos_x
    dist_dest_ball_y=dest_pos_y-ball_pos_y
    
    net_force_ikap=0
    net_force_jkap=0
    
    euclid_dist=euclid_distance(ball_pos_x,ball_pos_y,dest_pos_x,dest_pos_y)
    if(euclid_dist<300):
        net_force_ikap=-(k_rep_w_m*dist_dest_ball_x)
        net_force_jkap=-(k_rep_w_m*dist_dest_ball_y)

    net_force_ikap+=(k_att_w_m*dist_dest_ball_x)
    net_force_jkap+=(k_att_w_m*dist_dest_ball_y)

    net_force_ikap=net_force_ikap/max(1,abs(euclid_dist*(euclid_dist-40)))
    net_force_jkap=net_force_jkap/max(1,abs(euclid_dist*(euclid_dist-40)))

    return net_force_ikap,net_force_jkap


def find_net_force_wo_mask(window,canvas,index,balls,dest):
    
    net_force_ikap=0
    net_force_jkap=0

    ball_pos_x,ball_pos_y,t1,t2 = canvas.coords(balls[index])
    ball_pos_x=(int)(ball_pos_x+t1)/2
    ball_pos_y=(int)(ball_pos_y+t2)/2

    dest_pos_x,dest_pos_y,t1,t2 = canvas.coords(dest)
    dest_pos_x=(int)(dest_pos_x+t1)/2
    dest_pos_y=(int)(dest_pos_y+t2)/2

    dis_dest_ball_x=dest_pos_x-ball_pos_x
    dis_dest_ball_y=dest_pos_y-ball_pos_y

    euclid_dist=euclid_distance(ball_pos_x,ball_pos_y,dest_pos_x,dest_pos_y)

    net_force_ikap+=k_att*dis_dest_ball_x
    net_force_jkap+=k_att*dis_dest_ball_y
    
    if(euclid_dist>200):
        net_force_ikap=(net_force_ikap*200)/max(1,euclid_dist)
        net_force_jkap=(net_force_jkap*200)/max(1,euclid_dist)
        

    if(euclid_dist<40):
        net_force_ikap*=-1
        net_force_jkap*=-1
        
    for i in range(len(balls)):
        if(i!=index):
            t1,t2=find_same_force_wo_mask(window,canvas,balls[index],balls[i])
            net_force_ikap+=t1
            net_force_jkap+=t2

    return net_force_ikap,net_force_jkap

def find_net_force_w_mask(window,canvas,index,balls,dest):
    
    net_force_ikap=0
    net_force_jkap=0
    ball_pos_x,ball_pos_y,t1,t2 = canvas.coords(balls[index])
    ball_pos_x=(int)(ball_pos_x+t1)/2
    ball_pos_y=(int)(ball_pos_y+t2)/2

    dest_pos_x,dest_pos_y,t1,t2 = canvas.coords(dest)
    dest_pos_x=(int)(dest_pos_x+t1)/2
    dest_pos_y=(int)(dest_pos_y+t2)/2

    dis_dest_ball_x=dest_pos_x-ball_pos_x
    dis_dest_ball_y=dest_pos_y-ball_pos_y
    euclid_dist=euclid_distance(ball_pos_x,ball_pos_y,dest_pos_x,dest_pos_y)
    
    if(euclid_dist<250):
        net_force_ikap+=(-1*k_rep*dis_dest_ball_x)
        net_force_jkap+=(-1*k_rep*dis_dest_ball_y)
        net_force_ikap=net_force_ikap/max(1,abs(euclid_dist*(euclid_dist-40)*(euclid_dist-40)))
        net_force_jkap=net_force_jkap/max(1,abs(euclid_dist*(euclid_dist-40)*(euclid_dist-40)))
    
    
    for i in range(len(balls)):
        if(i!=index):
            t1,t2=find_same_force_w_mask(window,canvas,balls[index],balls[i])
            net_force_ikap+=t1
            net_force_jkap+=t2

    return net_force_ikap,net_force_jkap


def find_next_pos(window,canvas,index,balls,dest,label):
    if(label=="Mask"):
        net_force_dir_x,net_force_dir_y=find_net_force_w_mask(window,canvas,index,balls,dest)
    else:
        net_force_dir_x,net_force_dir_y=find_net_force_wo_mask(window,canvas,index,balls,dest)
    
    unit_force_dir_x=(net_force_dir_x/max(1,math.sqrt(pow(net_force_dir_x,2)+pow(net_force_dir_y,2))))
    
    unit_force_dir_y=(net_force_dir_y/max(1,math.sqrt(pow(net_force_dir_x,2)+pow(net_force_dir_y,2))))
    
    vel_x=(int)(vmax*unit_force_dir_x)
    
    vel_y=(int)(vmax*unit_force_dir_y)

    ball_pos_x,ball_pos_y,t1,t2 = canvas.coords(balls[index])
    ball_pos_x=(int)(ball_pos_x+t1)/2
    ball_pos_y=(int)(ball_pos_y+t2)/2
    
    if(ball_pos_x+vel_x<30 or ball_pos_x+vel_x>870):
        vel_x=0
    
    if(ball_pos_y+vel_y<30 or ball_pos_y+vel_y>645):
        vel_y=0
    
    return vel_x,vel_y



def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        preds = maskNet.predict(faces)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
    default="face_detector",
    help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
    default="mask_detector.model",
    help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
    "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
window = create_animation_window()
canvas = create_animation_canvas(window)
balls,dest=create_objects(window,canvas)
window.update()
time.sleep(2)
# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=900)
    frame = cv2.flip(frame,1)
    # (h, w) = frame.shape[:2]
    # print(h," ",w)
    # detect faces in the frame and determine if they are wearing a
    # face mask or not
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    # loop over the detected face locations and their corresponding
    # locations
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # determine the class label and color we'll use to draw
        # the bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        dest_pos_x=(startX+endX)/2
        dest_pos_y=(startY+endY)/2+(endY-startY)/4
        x1,y1,x2,y2=canvas.coords(dest)
        x1=(x1+x2)/2
        y1=(y1+y2)/2
        canvas.move(dest,dest_pos_x-x1,dest_pos_y-y1)
        #print("dest ",x1," ",y1)
        for i in range(len(balls)):
            xinc,yinc=find_next_pos(window,canvas,i,balls,dest,label)
            canvas.move(balls[i],xinc,yinc)
            x1,y1,x2,y2=canvas.coords(balls[i])
            x1=(int)((x1+x2)/2)
            y1=(int)((y1+y2)/2)
            #print("ball ",i," ",x1," ",y1)
        
            cv2.circle(frame,(x1,y1),20,(0,255,0),-1)
        window.update()
        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(frame, label, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    cv2.putText(frame,"GERMS SIMULATION", (300,40),cv2.FONT_HERSHEY_SIMPLEX, 1.2,(255,0,0), 3)
    for i in range(len(balls)):
        x1,y1,x2,y2=canvas.coords(balls[i])
        x1=(int)((x1+x2)/2)
        y1=(int)((y1+y2)/2)
        cv2.circle(frame,(x1,y1),20,(0,255,0),-1)
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()