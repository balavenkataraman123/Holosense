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
scale = 8/dispw
killed = 0
endist = 3
eedist = 4

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
        while killed == 0:
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
                if numinf % inference_interval == 0:
                    newtime = round(time.time() * 1000)
                    logs.append(str(newtime - starttime) + "," + str(e1x) + "," + str(e1y) + "," + str(e1z) + "\n")

                numinf += 1
                changed = 1
            except:
                pass
            cv2.putText(image, "User Position: " + str(int(e1x)) + "," + str(int(e1y)) + "," + str(int(e1z)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))
            cv2.imshow('preview', image)
            cv2.waitKey(1)
            

ktranslation = [0,-7,121]
verticies1 = (
    (-8, 5, 10),
    (8, 5, 10),
    (-8, -5, 10),
    (8, -5, 10),
    (-8, 5, 0),
    (8, 5, 0),
    (-8, -5, 0),
    (8, -5, 0)    
)

edges1 = (
    (0,1),
    (1,3),
    (0,2),
    (2,3),
    (4,5),
    (5,7),
    (4,6),
    (6,7),
    (0,4),
    (1,5),
    (2,6),
    (3,7)
    
)

def Cube():
    glBegin(GL_LINES)
    for edge in edges1:
        for vertex in edge:
            glColor3f(1, 0, 0)
            glVertex3fv(verticies1[vertex])


    glEnd()

def main():
    global killed
    global dilation
    global scale
    pygame.init()
    display = (dispw,disph)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
    thisdw = dispw*scale
    thisdh = disph*scale
    glFrustum(-thisdw*0.01, thisdw*0.01, -thisdh*0.01, thisdh*0.01, 10*0.01, 50.0)
    framenum = 0
    thisloc = [0,0,20]
    model = OBJ('katana/katana.obj')
    glTranslatef(0,0,-20)
    while True:
        #print(ktranslation)
        glLoadIdentity()
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)       

        distperinch = (((2*dispw*scale) ** 2 + (2*disph*scale)**2)**0.5)/screen_diagonal
        if changed == 0:
            glFrustum(-thisdw*0.01, thisdw*0.01, -thisdh*0.01, thisdh*0.01, 10*0.01, 50.0)
        else:
            thisloc = [(-e1x + camera_x_offset) * distperinch, distperinch*(e1y + camera_y_offset), (e1z + camera_z_offset)*distperinch] 
            # thisloc is relative to the dimension of the laptop scren - assuming it is 16 units wide and 9 units tall

            left = -thisdw - thisloc[0]
            right = thisdw - thisloc[0]
            top = thisdh - thisloc[1]
            bottom = -thisdh - thisloc[1]
            clipping = thisloc[2] - 10
            try:
                glFrustum(left*0.01, right*0.01, bottom*0.01, top*0.01, clipping*0.01, 150.0)
                glTranslatef(-thisloc[0],-thisloc[1],-thisloc[2])
            except:
                glFrustum(-thisdw*0.01, thisdw*0.01, -thisdh*0.01, thisdh*0.01, 10*0.01, 50.0)

        
        keypress = pygame.key.get_pressed()
        translation_mode = 0
        if translation_mode == 0:
            translated = 0
            if keypress[pygame.K_w]:
                ktranslation[2] -= 1
                translated = 1
            if keypress[pygame.K_s]:
                ktranslation[2] += 1
                translated = 1
            if keypress[pygame.K_d]:
                ktranslation[0] += 1
                translated = 1
            if keypress[pygame.K_a]:
                ktranslation[0] -= 1
                translated = 1
            if keypress[pygame.K_z]:
                ktranslation[1] -= 1
                translated = 1
            if keypress[pygame.K_x]:
                ktranslation[1] += 1
                translated = 1
            if keypress[pygame.K_j]:
                dilation *= 1.01
                translated = 1
            if keypress[pygame.K_l]:
                dilation /= 1.01
                translated = 1
        else:
            if keypress[pygame.K_w]:
                translation[2] -= 0.1
            if keypress[pygame.K_s]:
                translation[2] += 0.1
            if keypress[pygame.K_d]:
                translation[0] += 0.02
            if keypress[pygame.K_a]:
                translation[0] -= 0.02
            if keypress[pygame.K_z]:
                translation[1] -= 0.1
            if keypress[pygame.K_x]:
                translation[1] += 0.1
            if keypress[pygame.K_j]:
                dilation *= 1.01
            if keypress[pygame.K_l]:
                dilation /= 1.01            

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        Cube()

        glScalef(0.1 * dilation,0.1 * dilation,0.1 * dilation)
        glTranslate(ktranslation[0], ktranslation[1],ktranslation[2])        
        glPushMatrix()
        model.render()
        glPopMatrix()

        pygame.display.flip()
        pygame.time.wait(5)
    
th = threading.Thread(target=camerastuff)


th.start()

main()
