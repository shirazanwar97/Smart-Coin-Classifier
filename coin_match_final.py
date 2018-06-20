import numpy as np
import cv2
import RPi.GPIO as GPIO
from time import sleep
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
GPIO.setup(36,GPIO.OUT)
GPIO.setup(38,GPIO.OUT)
GPIO.setup(40,GPIO.OUT)

def servomotor_1():
        print('servo motor 1')
        GPIO.setup(3, GPIO.OUT)
        pwm=GPIO.PWM(3, 50)
        pwm.start(12.5)
        #while 1:                                                       # execute loop forever                                    
        #pwm.ChangeDutyCycle(7.5)                   # change duty cycle for getting the servo position to 90º
        sleep(30)                                      # sleep for 1 second
        '''pwm.ChangeDutyCycle(12.5)                  # change duty cycle for getting the servo position to 180º
                sleep(1)                                     # sleep for 1 second
        '''
        #sleep(3)
        pwm.ChangeDutyCycle(2.5)                  # change duty cycle for getting the servo position to 0º
        sleep(1) 
def servomotor_2():
        print('servo motor 2')
        GPIO.setup(5, GPIO.OUT)
        pwm=GPIO.PWM(5, 50)
        pwm.start(12.5)
        #while 1:                                                       # execute loop forever                                    
        #pwm.ChangeDutyCycle(7.5)                   # change duty cycle for getting the servo position to 90º
        sleep(30)                                      # sleep for 1 second
        '''pwm.ChangeDutyCycle(12.5)                  # change duty cycle for getting the servo position to 180º
                sleep(1)                                     # sleep for 1 second
        '''
        #sleep(3)
        pwm.ChangeDutyCycle(2.5)                  # change duty cycle for getting the servo position to 0º
        sleep(1) 

def servomotor_3():
        print('servo motor 3')
        GPIO.setup(15, GPIO.OUT)
        pwm=GPIO.PWM(15, 50)
        pwm.start(12.5)
        #while 1:                                                       # execute loop forever                                    
        #pwm.ChangeDutyCycle(7.5)                   # change duty cycle for getting the servo position to 90º
        sleep(30)                                      # sleep for 1 second
        '''pwm.ChangeDutyCycle(12.5)                  # change duty cycle for getting the servo position to 180º
                sleep(1)                                     # sleep for 1 second
        '''
        #sleep(3)
        pwm.ChangeDutyCycle(2.5)                  # change duty cycle for getting the servo position to 0º
        sleep(1) 
 
def servomotor_4():
        print('servo motor 4')
        GPIO.setup(13, GPIO.OUT)
        pwm=GPIO.PWM(13, 50)
        pwm.start(12.5)
        #while 1:                                                       # execute loop forever                                    
        #pwm.ChangeDutyCycle(7.5)                   # change duty cycle for getting the servo position to 90º
        sleep(30)                                      # sleep for 1 second
        '''pwm.ChangeDutyCycle(12.5)                  # change duty cycle for getting the servo position to 180º
                sleep(1)                                     # sleep for 1 second
        '''
        #sleep(3)
        pwm.ChangeDutyCycle(2.5)                  # change duty cycle for getting the servo position to 0º
        sleep(1) 


#import query image
query = cv2.imread('query.jpg',0) # queryImage

img=[cv2.imread('img1.jpg'),cv2.imread('5rs.jpg'),cv2.imread('1rs.jpg'),cv2.imread('2Rs.jpg')]
	
#declare array for size
num_match=[0,0,0,0]
# Initiate SIFT detector
orb = cv2.ORB()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(query,None)


kp2=[0,0,0,0]
des2=[0,0,0,0]
good_arr=[0,0,0,0]
for i in range(1,len(img)+1):
	kp2[i-1], des2[i-1] = orb.detectAndCompute(img[i-1],None)
	
	# create BFMatcher object
	bf = cv2.BFMatcher()

	matches = bf.knnMatch(des1,des2[i-1], k=2)

	# Apply ratio test
	good = []
	for m,n in matches:
	    if m.distance < 0.75*n.distance:
	        good.append([m])
	good_arr[i-1]=good
	num_match[i-1]=len(good)

	

temp=num_match
print(num_match)
max_match=max(temp)
#print(max_match)
indx=num_match.index(max_match)
print(indx)
#Draw first best matches.
#img3 = cv2.drawMatchesknn(query,kp1,img[indx],kp2[indx],good_arr[indx],None,flags=2)

#dc motor
GPIO.output(36,GPIO.HIGH)
GPIO.output(38,GPIO.LOW)
GPIO.output(40,GPIO.HIGH)
        
if indx==0:
        servomotor_1()
if indx==1:
        servomotor_2()
if indx==2:
        servomotor_3()
if indx==3:
        servomotor_4()

GPIO.output(36,GPIO.LOW)

#plt.imshow(img3),plt.show()
#cv2.imwrite('ploting',img3)
#cv2.imshow('image',img3)
#cv2.waitKey(0)
#cv2.destroyAllWindows()



