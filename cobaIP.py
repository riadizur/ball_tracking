import numpy as np
import cv2
import time
from datetime import datetime
#from matplotlib import pyplot as plt

#from imutils.video import WebcamVideoStream
#import imutils
################################################
###Variabel Konfigurasi Kamera
cap = cv2.VideoCapture(0)          #Sumber Kamera
#cap = WebcamVideoStream(src=1).start()
warna=""
Ndetect=6
detect=("kertas","tisu","cd","sms","kaleng","gelasplastik")
fWidth=360                        #Resolusi Frame Lebar
fHeight=240                        #Resolusi Frame Tinggi
ret=cap.set(3,fWidth)              
ret=cap.set(4,fHeight)

###############################################
###Variabel umum
img_counter=0
tampil=0
maxtampil=6          #sesuaikan dengan banyaknya tampilan
                     #yang diinginkan
show=True
tunning=False
serial=False
line=False
circle=False
cirDet=False
lineDet=False
convexDet=False
hullDet=False
printVal=False
dilasi=False
running=False
mode=0
maxMode=50
###########################################################
##Kode yang akan dikirimkan melalui serial
#code=(('a','b','c','d','1'),('e','f','g','h','2'),('i','j','k','l','3'),('4','5','6','7','8'),('A','B','C','D','E')) 
code=(('a','b'),('c','d'),('e','f')) 

######################################################
##Program serial


#####################################################
##Program Trackbar
def nothing(x):
	pass
    


def exportTrackbarsValue1():
    global hfmax, hfmin, sfmax, sfmin, vfmax, vfmin, efsize, dfsize
    hfmax = cv2.getTrackbarPos('HMax','Control1')
    hfmin = cv2.getTrackbarPos('HMin','Control1')
    sfmax = cv2.getTrackbarPos('SMax','Control1')
    sfmin = cv2.getTrackbarPos('SMin','Control1')
    vfmax = cv2.getTrackbarPos('VMax','Control1')
    vfmin = cv2.getTrackbarPos('VMin','Control1')
    efsize = cv2.getTrackbarPos('Erode','Control1')
    dfsize = cv2.getTrackbarPos('Dilate','Control1')
    return None

def exportTrackbarsValue2():     
    global hfmax1, hfmin1, sfmax1, sfmin1, vfmax1, vfmin1, efsize1, dfsize1
    hfmax1 = cv2.getTrackbarPos('HMax1','Control2')
    hfmin1 = cv2.getTrackbarPos('HMin1','Control2')
    sfmax1 = cv2.getTrackbarPos('SMax1','Control2')
    sfmin1 = cv2.getTrackbarPos('SMin1','Control2')
    vfmax1 = cv2.getTrackbarPos('VMax1','Control2')
    vfmin1 = cv2.getTrackbarPos('VMin1','Control2')
    efsize1 = cv2.getTrackbarPos('Erode1','Control2')
    dfsize1 = cv2.getTrackbarPos('Dilate1','Control2')
    return None

def loadConfig():
    try:
        f = open("storageThreshold-res%dx%d-sampah-%s.txt"%(fWidth,fHeight,warna),"r")
    except:
        f = open("storageThreshold-res%dx%d-sampah-%s.txt"%(fWidth,fHeight,warna),"w+")
        data = '%d,%d,%d,%d,%d,%d,%d,%d'%(0,0,0,0,0,0,0,0)
        data1 = '%d,%d,%d,%d,%d,%d,%d,%d'%(0,0,0,0,0,0,0,0)
        data2=data+','+data1
        print (data2)
        f.write(data2)
        f.close()
        print 'saved'
        f = open("storageThreshold-res%dx%d-sampah-%s.txt"%(fWidth,fHeight,warna),"r")
    global hfmax, hfmin, sfmax, sfmin, vfmax, vfmin, efsize, dfsize
    global hfmax1, hfmin1, sfmax1, sfmin1, vfmax1, vfmin1, efsize1, dfsize1
    for line in f.readlines():
        arr_read = line.split(',')
        hfmax = int(arr_read[0])
        hfmin = int(arr_read[1])
        sfmax = int(arr_read[2])
        sfmin = int(arr_read[3])
        vfmax = int(arr_read[4])
        vfmin = int(arr_read[5])
        efsize = int(arr_read[6])
        dfsize = int(arr_read[7])
        hfmax1 = int(arr_read[8])
        hfmin1 = int(arr_read[9])
        sfmax1 = int(arr_read[10])
        sfmin1 = int(arr_read[11])
        vfmax1 = int(arr_read[12])
        vfmin1 = int(arr_read[13])
        efsize1 = int(arr_read[14])
        dfsize1 = int(arr_read[15])
    print 'loaded'
    return None

def loadTrackbars1():
    global hfmax, hfmin, sfmax, sfmin, vfmax, vfmin, efsize, dfsize
    loadConfig()
    cv2.setTrackbarPos('HMax','Control1',hfmax)
    cv2.setTrackbarPos('HMin','Control1',hfmin)
    cv2.setTrackbarPos('SMax','Control1',sfmax)
    cv2.setTrackbarPos('SMin','Control1',sfmin)
    cv2.setTrackbarPos('VMax','Control1',vfmax)
    cv2.setTrackbarPos('VMin','Control1',vfmin)
    cv2.setTrackbarPos('Erode','Control1',efsize)
    cv2.setTrackbarPos('Dilate','Control1',dfsize)
    return None

def loadTrackbars2():
    global hfmax1, hfmin1, sfmax1, sfmin1, vfmax1, vfmin1, efsize1, dfsize1
    loadConfig()
    cv2.setTrackbarPos('HMax1','Control2',hfmax1)
    cv2.setTrackbarPos('HMin1','Control2',hfmin1)
    cv2.setTrackbarPos('SMax1','Control2',sfmax1)
    cv2.setTrackbarPos('SMin1','Control2',sfmin1)
    cv2.setTrackbarPos('VMax1','Control2',vfmax1)
    cv2.setTrackbarPos('VMin1','Control2',vfmin1)
    cv2.setTrackbarPos('Erode1','Control2',efsize1)
    cv2.setTrackbarPos('Dilate1','Control2',dfsize1)
    return None

def createTrackbars1():
    cv2.namedWindow('Control1')
    cv2.createTrackbar('HMax','Control1',255,255,nothing)
    cv2.createTrackbar('HMin','Control1',0,255,nothing)
    cv2.createTrackbar('SMax','Control1',255,255,nothing)
    cv2.createTrackbar('SMin','Control1',0,255,nothing)
    cv2.createTrackbar('VMax','Control1',255,255,nothing)
    cv2.createTrackbar('VMin','Control1',0,255,nothing)
    cv2.createTrackbar('Erode','Control1',0,10,nothing)
    cv2.createTrackbar('Dilate','Control1',0,100,nothing)
    #loadTrackbars1()
    return None

def createTrackbars2():
    cv2.namedWindow('Control2')
    cv2.createTrackbar('HMax1','Control2',255,255,nothing)
    cv2.createTrackbar('HMin1','Control2',0,255,nothing)
    cv2.createTrackbar('SMax1','Control2',255,255,nothing)
    cv2.createTrackbar('SMin1','Control2',0,255,nothing)
    cv2.createTrackbar('VMax1','Control2',255,255,nothing)
    cv2.createTrackbar('VMin1','Control2',0,255,nothing)
    cv2.createTrackbar('Erode1','Control2',0,10,nothing)
    cv2.createTrackbar('Dilate1','Control2',0,100,nothing)
    loadTrackbars2()
    return None

###########################################################################
###### Program menyimpan dan meload konfigurasi pengolahan Citra
def saveConfig():
    warna=raw_input("Masukkan Nama Sampah :")
    try:
        f = open("storageThreshold-res%dx%d-sampah-%s.txt"%(fWidth,fHeight,warna),"w")
    except:
        f = open("storageThreshold-res%dx%d-sampah-%s.txt"%(fWidth,fHeight,warna),"w+")
    data = '%d,%d,%d,%d,%d,%d,%d,%d'%(hfmax,hfmin,sfmax,sfmin,vfmax,vfmin,efsize,dfsize)
    data1 = '%d,%d,%d,%d,%d,%d,%d,%d'%(hfmax1,hfmin1,sfmax1,sfmin1,vfmax1,vfmin1,efsize1,dfsize1)
    data2=data+','+data1
    print (data2)
    f.write(data2)
    f.close()
    print 'saved'
    return None

########################################################################
### Program Menjejakkan trace objek
from collections import deque
pts = deque(maxlen=int(64))
def garis(center):
    pts.appendleft(center)
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:    
            continue

        thickness = int(np.sqrt(int(64) / float(i + 1)) * 2.5)
        #print thickness
        cv2.line(image, pts[i - 1], pts[i], (0, 0, 255), thickness)

bunyi=0
def object_detection(parsefield):
    cnts= cv2.findContours(parsefield.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None
    global bunyi
    if len(cnts) > 5:
        hull=[]
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        try:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        except:
            pass
        if printVal==True:
                print len(cnts)
        if convexDet==True or hullDet==True:
                for i in range(len(cnts)):
                        hull.append(cv2.convexHull(cnts[i],False))
                        color_contours=(0,255,0)
                        color=(255,0,0)
                        if convexDet==True:
                                cv2.drawContours(image,cnts,i,color_contours,2,8)
                        if hullDet==True:
                                cv2.drawContours(image,hull,i,color,2,8)
        #print radius
        if radius > 15:
            if circle==True:
                cv2.circle(image, (int(x), int(y)), int(radius),(0, 255, 255), 2)
                cv2.circle(image, center, 5, (0, 0, 255), -1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image,'%s'%warna,(int(x), int(y)),font,1,(0,0,0),2)

            if line==True:
                garis(center)               #Menampilkan jejak trace object
            
            kirim=''
            if center is None:
                str_x, str_y = "x---", "y---"
            else:
                eex,yee=0,0
                str_x, str_y = "x={:04d}".format(center[0]), "y={:04d}".format(center[1])
                ex,ye=center[0],center[1]
                eex=ex/(fWidth/2)
                yee=ye/(fHeight/3)
                for i in range(3):
                    for j in range(2):
                        if eex==j and yee==i:
                            kirim=code[i][j]
                if serial==True and kirim!='':
                        sendData(kirim)            #Mengirim data melalui serial
                else:
                        print kirim
                        if running==True:
                                global imagee
                                global img_counter
                                seconds=1545925769.9618232
                                waktu=datetime.now()
                                img_name="/home/pi/mts3pku/{}/{}.png".format(warna,waktu)
                                cv2.imwrite(img_name,imagee)
                                print ("{} written Done !".format(img_name))
                #print str_x, str_y
                #print eex,yee
        else:
                if serial==True:
                        sendData('X')            #Mengirim data melalui serial
                else:
                        print ('X')
                        bunyi=0
    else:
            if serial==True:
                    sendData('X')
                    bunyi=0
                    #print ('X')


def line_detection(edge):
    lines = cv2.HoughLines(edge,1,np.pi/180,50)
    if lines is not None:
        for rho,theta in lines[0]:
            a=np.cos(theta)
            b=np.sin(theta)
            x0=a*rho
            y0=b*rho
            x1=int(x0+1000*(-b))
            y1=int(y0+1000*(a))
            x2=int(x0-1000*(-b))
            y2=int(y0-1000*(a))
            cv2.line(image,(x1,y1),(x2,y2),(0,0,255),2)

def circle_detection(edge):
    #circles = cv2.HoughCircles(edge,cv2.HOUGH_GRADIENT,2,1,128,100,200)
    circles = cv2.HoughCircles(edge,cv2.HOUGH_GRADIENT,3,600)
    if circles is not None:
        circles=np.round(circles[0,:]).astype("int")
        for (x,y,r) in circles:
            cv2.circle(image, (x,y), r,(0, 255, 0), 4)
            #cv2.circle(image, center, 5, (0, 0,t 255), -1)

def mulObject_detection(objects,background):
    #cnts= cv2.findContours(objects.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnts1= cv2.findContours(background.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None
    if len(cnts1) > 15:
        hull=[]
        c = max(cnts1, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        try:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        except:
            pass
        if convexDet==True or hullDet==True:
                for i in range(len(cnts1)):
                        hull.append(cv2.convexHull(cnts1[i],False))
                        color_contours=(0,255,0)
                        color=(255,0,0)
                        if convexDet==True:
                                cv2.drawContours(image,cnts1,i,color_contours,2,8)
                        if hullDet==True:
                                cv2.drawContours(image,hull,i,color,2,8)

#########################################################################
###Program Utama
import pygame
import time
loadConfig()
counttt=0
while(True):
    counttt=counttt+1
    global warna
    if counttt>Ndetect-1:
            counttt=0
    warna=detect[counttt]
    if tunning == True:
            warna="baru"
    loadConfig()
    if tunning==True:
            exportTrackbarsValue1()
            exportTrackbarsValue2()
    ret, image = cap.read()
    global imagee
    imagee=image
    try:
            en=mode*2+1
            blur=cv2.GaussianBlur(image,(en,en),0)
            img=image.copy()
            image=blur.copy()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except:
            print ('Changing Camera source 0')
            cap = cv2.VideoCapture(0)
            #cap = WebcamVideoStream(src=0).start()
            image = cap.read()
            en=mode*2+1
            blur=cv2.GaussianBlur(image,(en,en),0)
            img=image.copy()
            image=blur.copy()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    hsv_blur = cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
    f_lower_val = np.array([hfmin,sfmin,vfmin])
    f_upper_val = np.array([hfmax,sfmax,vfmax])
    f_lower_val1 = np.array([hfmin1,sfmin1,vfmin1])
    f_upper_val1 = np.array([hfmax1,sfmax1,vfmax1])
    parsefield = cv2.inRange(hsv_blur,f_lower_val,f_upper_val)
    parsefield1 = cv2.inRange(hsv_blur,f_lower_val1,f_upper_val1)
    #parsefield1= cv2.bitwise_not(parsefield1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    erosion = cv2.erode(parsefield,kernel,iterations = efsize)
    erosion1 = cv2.erode(parsefield1,kernel,iterations = efsize1)
    dilate = cv2.dilate(erosion, kernel, iterations = dfsize)
    dilate1 = cv2.dilate(erosion1, kernel, iterations = dfsize1)
    onbackground=cv2.bitwise_and(dilate,dilate1)
    edge = cv2.Canny(dilate,100,200)
    image=img.copy()
    output=onbackground
    object_detection(onbackground)
    #mulObject_detection(parsefield,parsefield1)
    
    if lineDet==True:
            line_detection(output)
    if cirDet==True:
            circle_detection(output)

    if show==True:
            if tampil==0:
                cv2.imshow('Original',image)
            elif tampil==1:
                cv2.imshow('gray',gray)
            elif tampil==2:
                cv2.imshow('tepi',edge)
            elif tampil==3:
                cv2.imshow('hsv',hsv)
            elif tampil==4:
                cv2.imshow('hsv_blur',hsv_blur)
            elif tampil==5:
                cv2.imshow('parsefield',parsefield)
                cv2.imshow('parsefield1',parsefield1)
                cv2.imshow('On Background',onbackground)
            elif tampil==6:
                cv2.imshow('Erosion',erosion)
            elif tampil==7:
                cv2.imshow('dilate',dilate)
                
    k=cv2.waitKey(1)
    if k%256 == 32:
        img_name="/home/pi/mts3pku/database/capture{}.png".format(img_counter)
        cv2.imwrite(img_name,gray)
        print ("{} written Done !".format(img_name))
        img_counter=img_counter+1
    elif k%256 == ord('q'):
        print ("Closing Application...")
        break
    elif k%256== ord('k'):
        tunning=True
        createTrackbars1()
        createTrackbars2()
        loadConfig()
        loadTrackbars1()
        loadTrackbars2()
    elif k%256== ord('s'):
        saveConfig()
    elif k%256==ord('t'):
        tampil=tampil+1
        tunning=False
        if tampil>maxtampil:
            tampil=0
        cv2.destroyAllWindows()
    elif k%256==ord('r'):
        tampil=tampil-1
        tunning=False
        if tampil<0:
            tampil=maxtampil
        cv2.destroyAllWindows()
    elif k%256==ord('m'):
        mode=mode+1
        if mode>maxMode:
            mode=0
    elif k%256==ord('n'):
        mode=mode-1
        if mode<0:
            mode=maxMode
    elif k%256==ord('g'):
        if line==False:
            line=True
        else:
            line=False
    elif k%256==ord('j'):
        if lineDet==False:
            lineDet=True
        else:
            lineDet=False
    elif k%256==ord('l'):
        if cirDet==False:
            cirDet=True
        else:
            cirDet=False
    elif k%256==ord('i'):
        if dilasi==False:
            dilasi=True
        else:
            dilasi=False
    elif k%256==ord('o'):
        if printVal==False:
            printVal=True
        else:
            printVal=False
    elif k%256==ord('p'):
        serial=True
        '''
        if show==False:
            show=True
        else:
            show=False
        cv2.destroyAllWindows()'''
    elif k%256==ord('v'):
        if convexDet==False:
            convexDet=True
        else:
            convexDet=False
    elif k%256==ord('h'):
        if hullDet==False:
            hullDet=True
        else:
            hullDet=False
    elif k%256==ord('d'):
        if serial==False:
            serial=True
        else:
            serial=False
    elif k%256==ord('c'):
        if circle==False:
            circle=True
        else:
            circle=False
    elif k%256==ord('0'):
        running=True
        print('Lampu Merah...!')
        #program suara
        pygame.mixer.init()
        pygame.mixer.music.load("/home/pi/TA_Dayu/Suara/lampumerah.mp3")
        pygame.mixer.music.play()
        time.sleep(3.5)
    elif k%256==ord('1'):
        running=False
        print('Lampu Hijau...!')
        #program suara
        pygame.mixer.init()
        pygame.mixer.music.load("/home/pi/TA_Dayu/Suara/lampuhijau.mp3")
        pygame.mixer.music.play()
        time.sleep(3.5)
    elif k%256==ord('2'):
        running=False
        print('Lampu Kuning...!')
        #program suara
        pygame.mixer.init()
        pygame.mixer.music.load("/home/pi/TA_Dayu/Suara/lampukuning.mp3")
        pygame.mixer.music.play()
        time.sleep(3.5)
    else:
        if k%256!=255:
            print('Parameter Salah !')
            
cap.release()
#cap.stop()
cv2.destroyAllWindows()
