import mediapipe as mp
import threading
import math
import cv2
#from objloader import *
import time
import configdata
import serial

ser = serial.Serial("/tmp/ttyBLE")

thisconf = configdata.config
e1x = 0
e1y = 0
e1z = 0
changed = 0

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

dilation = 1

camw = thisconf.camw
camh = thisconf.camh
HFOV = thisconf.HFOV * math.pi / 180
VFOV = thisconf.VFOV * math.pi / 180
htan = - math.tan(HFOV/2)
vtan =  math.tan(VFOV/2)
dispw = thisconf.dispw
disph = thisconf.disph
screen_diagonal = thisconf.screen_diagonal # measurement is in inches
camera_y_offset = thisconf.camera_y_offset # vertical distance between the center of the screen and the center of the camera lens in inches
camera_x_offset = thisconf.camera_x_offset
camera_z_offset = thisconf.camera_z_offset
scale = 8/dispw
killed = 0
endist = thisconf.endist
eedist = thisconf.eedist

cap = cv2.VideoCapture(thisconf.cameraname)
thisconf.configcamera(thisconf, cap)
cam_inference_interval = thisconf.camera_inference_interval
dist_inference_interval = thisconf.dsensor_inference_interval

lfp = "cameradistance1"
dfp = "sensordistance1"

logs = []
distlogs = []


starttime = 0
starttime2 = round(time.time() * 1000)

time.sleep(3)

numinf = 0
def eval_func(a, consts):
    [c1, c2, c3, c4, c5, c6] = consts
    bp = func_b(c1, c2, a)
    cp = func_b(c3, c4, a)
    return [(bp**2 + cp**2) - (2*cp*bp*c5) - (c6)**2, bp, cp]
def func_b(c1, c2, a):
    return ((2*a*c1) + ((2*a*c1) ** 2 + 4*(c2 **2 - a**2))**0.5) / 2
def nr_approx(first, last, step, consts):
    zeroes = []
    [zfirst, bd, cd] = eval_func(first, consts)
    for j in range(int((last - first) / step)):
        a = (j * step) + first
        [ff, bd,cd] = eval_func(a, consts)
        try:
            zz = int(ff)
        except:
            ff = zfirst * -1
        if ff > 0 and zfirst < 0 or ff < 0 and zfirst > 0:
            return [a, bd.real, cd.real]
lex = 0
ley = 0
rex = 0
rey = 0
nx = 0
ny = 0
currentlog = 0

while True:
    serline = ser.readline()
    print(serline.decode())
    print("awaiting confirmation")
    if serline.decode().rstrip() == "GOGOGOGOGO":
        break
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
    while True: #(round(time.time() * 1000) - starttime2) < 8000:
        serline = ser.readline().decode().strip()
        print(serline)
        if serline == "ENDITRIGHTNOW":
            print("writing the log file")
            f = open(str(currentlog) + lfp, "a")
            for i in logs:
                f.write(i)
            f.close()                  
            currentlog += 1
            logs = []
            while True:
                serline = ser.readline().decode().strip()
                if serline == "GOGOGOGOGO":
                    break
            continue
        
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            print("detection")

            
            for face_landmarks in results.multi_face_landmarks:
                #print(face_landmarks)
                xcoords = [handmark.x for handmark in face_landmarks.landmark]
                ycoords = [handmark.y for handmark in face_landmarks.landmark]
                
                
                rex = (xcoords[33] * camw)
                rey = (ycoords[33] * camh)
                
                lex = (xcoords[263] * camw)
                ley = (ycoords[263] * camh)

                nx = (xcoords[19] * camw)
                ny = (ycoords[19] * camh)
                cv2.circle(image, (int(rex), int(rey)), 3, (255, 0, 0), -1)
                cv2.circle(image, (int(lex), int(ley)), 3, (255, 0, 0), -1)
                cv2.circle(image, (int(nx), int(ny)), 3, (255, 0, 0), -1)

        
        e1h = htan * ((camw/2) - rex)/640 
        e1v = vtan * ((camh/2) - rey)/360
        e2h = htan * ((camw/2) - lex)/640
        e2v = vtan * ((camh/2) - ley)/360
        nh = htan * ((camw/2) - nx)/640
        nv = vtan * ((camh/2) - ny)/360
        # angle cosine calculation by dot products 
        eec = ((e1h * e2h) + (e1v * e2v) + 1) / (((e1h ** 2 + e1v ** 2 + 1) ** 0.5) * ((e2h ** 2 + e2v ** 2 + 1) ** 0.5))
        en1 = ((e1h * nh) + (e1v * nv) + 1) / (((e1h ** 2 + e1v ** 2 + 1) ** 0.5) * ((nh ** 2 + nv ** 2 + 1) ** 0.5))
        en2 = ((e2h * nh) + (e2v * nv) + 1) / (((e2h ** 2 + e2v ** 2 + 1) ** 0.5) * ((nh ** 2 + nv ** 2 + 1) ** 0.5))
        #print("distance approximation")
        try:
            [nd1, led, red] = nr_approx(0,90,1,[en2, endist, en1, endist, eec, eedist])
            [nd, led, red] = nr_approx(nd1-1,nd1,0.01,[en2, endist, en1, endist, eec, eedist])
            #print("success")
            rer = (red / ((e1h ** 2 + e1v ** 2 + 1)**0.5))
            ler = (led / ((e2h ** 2 + e2v ** 2 + 1)**0.5))
            prc = [(e1h * rer + e2h * ler)/2, (e1v * rer + e2v * ler)/2, (ler + rer)/2]
            e1x = prc[0]
            e1y = prc[1]
            e1z = prc[2]
            
            cv2.putText(image, "User Position (in inches relative to camera): " + str(round(e1x, 3)) + "," + str(round(e1y, 3)) + "," + str(round(e1z, 3)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))            
            newtime = round(time.time() * 1000)
            logs.append(str(newtime - starttime) + "," +str(e1x * 2.54) + "," +str(e1y * 2.54) + "," +  str(e1z * 2.54) + "," + serline + "\n")
            changed = 1
        except:
            pass
        cv2.imshow("viewfinder", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break        
