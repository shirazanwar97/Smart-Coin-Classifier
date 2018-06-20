import numpy as np
import cv2

#import query image
query = cv2.imread('query.jpg',0) # queryImage

img=[cv2.imread('5rs.jpg'),cv2.imread('img1.jpg'),cv2.imread('2Rs.jpg'),cv2.imread('1rs.jpg')]
	
#declare array for size
num_match=[0,0,0,0]
# Initiate SIFT detector
orb = cv2.ORB_create()

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
print(max_match)
indx=num_match.index(max_match)
print(indx)
#Draw first best matches.
img3 = cv2.drawMatchesKnn(query,kp1,img[indx],kp2[indx],good_arr[indx],None,flags=2)



'''if indx==0:
        servomotor1()
if indx==1:
        servomotor2()
if indx=2
        servomotor3()
if indx=3
        servomotor4()
'''
#plt.imshow(img3),plt.show()
#cv2.imwrite('ploting',img3)
cv2.imshow('image',img3)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
def servomotor1():
        import RPi.GPIO as GPIO
        from time import sleep
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(3, GPIO.OUT)
        pwm=GPIO.PWM(3, 50)
        pwm.start(0)
        def SetAngle(angle):
                duty = angle / 18 + 2
                GPIO.output(3, True)
                pwm.ChangeDutyCycle(duty)
                sleep(1)
                GPIO.output(3, False)
                pwm.ChangeDutyCycle(0)
        SetAngle(90)
def servomotor2():
        import RPi.GPIO as GPIO
        from time import sleep
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(3, GPIO.OUT)
        pwm=GPIO.PWM(3, 50)
        pwm.start(0)
        def SetAngle(angle):
                duty = angle / 18 + 2
                GPIO.output(3, True)
                pwm.ChangeDutyCycle(duty)
                sleep(1)
                GPIO.output(3, False)
                pwm.ChangeDutyCycle(0)
        SetAngle(90)
'''

# https://www.hongweipeng.com/index.php/archives/709/
# https://stackoverflow.com/questions/20259025/module-object-has-no-attribute-drawmatches-opencv-python
#def drawMatchesknn(img1, kp1, img2, kp2, matches):
'''
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated
    keypoints, as well as a list of DMatch data structure (matches)
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching 

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1, :cols1] = np.dstack([img1])

    # Place the next image to the right of it
    out[:rows2, cols1:] = np.dstack([img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)

    # Show the image
    # cv2.imshow('Matched Features', out)
    # cv2.waitKey(0)
    # cv2.destroyWindow('Matched Features')

    # Also return the image if you'd like a copy
    return out
'''
