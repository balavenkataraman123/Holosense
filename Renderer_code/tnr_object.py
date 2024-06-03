import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import mediapipe as mp
import threading
import math
import cv2
from objloader import *
import time
import configdata

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
endist = thisconf.endist
eedist = thisconf.eedist

cap = cv2.VideoCapture(thisconf.cameraname)
thisconf.configcamera(thisconf, cap)
inference_interval = thisconf.inference_interval
lfp = thisconf.log_file
logs = []
starttime = round(time.time() * 1000)



def camerastuff(): # VIDEO ANALYSIS - GETS FACIAL COORDINATES AND DOES SPICY MATH TO FIND SPATIAL POSITION OF POV
    global logs
    global changed
    global e1x
    global e1y
    global e1z
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
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        while True:
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                #print("detection")

                
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

            
            e1h = htan * ((camw/2) - rex)/(camw/2) 
            e1v = vtan * ((camh/2) - rey)/(camh/2)
            e2h = htan * ((camw/2) - lex)/(camw/2)
            e2v = vtan * ((camh/2) - ley)/(camh/2)
            nh = htan * ((camw/2) - nx)/(camw/2)
            nv = vtan * ((camh/2) - ny)/(camh/2)
            # angle cosine calculation by dot products 
            eec = ((e1h * e2h) + (e1v * e2v) + 1) / (((e1h ** 2 + e1v ** 2 + 1) ** 0.5) * ((e2h ** 2 + e2v ** 2 + 1) ** 0.5))
            en1 = ((e1h * nh) + (e1v * nv) + 1) / (((e1h ** 2 + e1v ** 2 + 1) ** 0.5) * ((nh ** 2 + nv ** 2 + 1) ** 0.5))
            en2 = ((e2h * nh) + (e2v * nv) + 1) / (((e2h ** 2 + e2v ** 2 + 1) ** 0.5) * ((nh ** 2 + nv ** 2 + 1) ** 0.5))
            #print("distance approximation")
            try:
                [nd1, led, red] = nr_approx(0,90,1,[en2, endist, en1, endist, eec, eedist])
                [nd, led, red] = nr_approx(nd1-1,nd1,0.01,[en2, endist, en1, endist, eec, eedist])
                rer = (red / ((e1h ** 2 + e1v ** 2 + 1)**0.5))
                ler = (led / ((e2h ** 2 + e2v ** 2 + 1)**0.5))
                prc = [(e1h * rer + e2h * ler)/2, (e1v * rer + e2v * ler)/2, (ler + rer)/2]
                e1x = prc[0]
                e1y = prc[1]
                e1z = prc[2]
                if numinf % inference_interval == 0:
                    newtime = round(time.time() * 1000)
                    logs.append(str(newtime - starttime) + "," + str(e1x) + "," + str(e1y) + "," + str(e1z) + "\n")

                numinf += 1
                changed = 1
            except:
                pass
            cv2.putText(image, "User Position: " + str(round(e1x, 3)) + "," + str(round(e1y, 3)) + "," + str(round(e1z, 3)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))
            cv2.imshow('preview', image)
            cv2.waitKey(1)
            

ktranslation = [0,0,0]

def main():
    # viewport coordinates

    top = 0
    left = 0
    right = 3840
    bottom = 2400
    

    pygame.init()
    display = (dispw,disph)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
    model = OBJ('katana/katana.obj')
    object_length_irl = 20 # how long the object will appear, in inches.
    object_length_gamespace = 259 # size in the game files.
    world2game = object_length_gamespace/object_length_irl
    monitor_height = screen_diagonal * disph / (dispw ** 2 + disph ** 2)**0.5
    monitor_width = screen_diagonal * dispw / (dispw ** 2 + disph ** 2)**0.5
    print("monitor width: " + str(monitor_width) + " inches.")
    thisloc = [0,0,20*world2game]

    while True:
        glLoadIdentity()
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)       

        

        thisloc = [(e1x + camera_x_offset)*world2game , -(e1y + camera_y_offset)* world2game, (e1z + camera_z_offset) * world2game] # eye's position rel. to the screen's position, in game space coordinates
        left = thisloc[0] - (monitor_width/2 * world2game)
        right = thisloc[0] + (monitor_width/2 * world2game)
        top = thisloc[1] + (monitor_height/2 * world2game)
        bottom = thisloc[1] - (monitor_height/2 * world2game)
        near = thisloc[2]
        glFrustum(left, right, bottom, top, near, 400.0)
        glTranslatef(thisloc[0],thisloc[1],-thisloc[2]) # makes the eye location in the view matrix to the eye location in real life
            
        
        keypress = pygame.key.get_pressed()
        if keypress[pygame.K_w]:
            ktranslation[2] -= 1
        if keypress[pygame.K_s]:
            ktranslation[2] += 1
        if keypress[pygame.K_d]:
            ktranslation[0] += 1
        if keypress[pygame.K_a]:
            ktranslation[0] -= 1
        if keypress[pygame.K_z]:
            ktranslation[1] -= 1
        if keypress[pygame.K_x]:
            ktranslation[1] += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        glTranslate(ktranslation[0], ktranslation[1],ktranslation[2])# moves the object relative to the screen position        
        
        glPushMatrix() 
        model.render()
        glPopMatrix()

        pygame.display.flip()
        pygame.time.wait(5)
    
th = threading.Thread(target=camerastuff)


th.start()

main()
