import numpy as np
import cv2

img1 = cv2.imread('query.jpg',0)          # queryImage
img2 = cv2.imread('train.jpeg',0) # trainImage

# Initiate SIFT detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
#bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
bf = cv2.BFMatcher()
# Match descriptors.
#matches = bf.match(des1,des2)

# Sort them in the order of their distance.
#matches = sorted(matches, key = lambda x:x.distance)

matches = bf.knnMatch(des1,des2, k=2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

# Draw first 10 matches.
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)

#plt.imshow(img3),plt.show()
#cv2.imwrite('ploting',img3)
cv2.imshow('image',img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
