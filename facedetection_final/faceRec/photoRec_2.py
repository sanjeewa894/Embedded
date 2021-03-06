import sys
import cv2
import cv
import Image, math
import os
from cv2 import *

#import RPi.GPIO as GPIO
import RPi.GPIO as GPIO

import time

# append tinyfacerec to module search path
sys.path.append("..")
# import numpy and matplotlib colormaps
import numpy as np
# import tinyfacerec modules
from tinyfacerec.util import read_images,normalize #class in tinyfacerec/util.py
from tinyfacerec.model import EigenfacesModel #class in tinyfacerec/model.py
# from tinyfacerec.crop_face import CropFace

def read_images(path, sz=None):

	X,y = [], []
	for dirname, dirnames, filenames in os.walk(path):
		#for subdirname in dirnames:
			#subject_path = os.path.join(dirname, subdirname)
			for filename in filenames:
			#for filename in os.listdir(subject_path): #listdir returns a list of all the entries in the path subject_path
				try:
					im = Image.open(os.path.join(dirname, filename)) #(os.path.join: joining paths (concatenating) intellgently)
					im = im.convert("L") #Convert to monochrome
					# resize to given size (if given)
					if (sz is not None):
						im = im.resize(sz, Image.ANTIALIAS)
					X.append(np.asarray(im, dtype=np.uint8)) #np.asarray:returns an array from im
					folders = dirname.split("/")
					y.append(int(folders[len(folders)-2]))  #To get the name of the batch
				except IOError:
					print "I/O error({0}): {1}".format(errno, strerror)
				except:
					print "Unexpected error:", sys.exc_info()[0]
					raise
			
	return [X,y]



def Distance(p1,p2):
	dx = p2[0] - p1[0]
	dy = p2[1] - p1[1]
	return math.sqrt(dx*dx+dy*dy)

def ScaleRotateTranslate(image, angle, center = None, new_center = None, scale = None, resample=Image.BICUBIC):
	if (scale is None) and (center is None):
		return image.rotate(angle=angle, resample=resample)
	nx,ny = x,y = center
		
	sx=sy=1.0
	if new_center:
		(nx,ny) = new_center
	if scale:
		(sx,sy) = (scale, scale)
	cosine = math.cos(angle)
	sine = math.sin(angle)
	a = cosine/sx
	b = sine/sx
	c = x-nx*a-ny*b
	d = -sine/sy
	e = cosine/sy
	f = y-nx*d-ny*e
	return image.transform(image.size, Image.AFFINE, (a,b,c,d,e,f), resample=resample)

def CropFace(image, eye_left=(0,0), eye_right=(0,0), offset_pct=(0.2,0.2), dest_sz = (70,70)):
  # calculate offsets in original image
	offset_h = math.floor(float(offset_pct[0])*dest_sz[0])
	offset_v = math.floor(float(offset_pct[1])*dest_sz[1])
  # get the direction
	eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1])
  # calc rotation angle in radians
	rotation = -math.atan2(float(eye_direction[1]),float(eye_direction[0]))
  # distance between them
	dist = Distance(eye_left, eye_right)
  # calculate the reference eye-width
	reference = dest_sz[0] - 2.0*offset_h
  # scale factor
	scale = float(dist)/float(reference)
  # rotate original around the left eye
	image = ScaleRotateTranslate(image, center=eye_left, angle=rotation)
  # crop the rotated image
	crop_xy = (eye_left[0] - scale*offset_h, eye_left[1] - scale*offset_v)
	crop_size = (dest_sz[0]*scale, dest_sz[1]*scale)
	image = image.crop((int(crop_xy[0]), int(crop_xy[1]), int(crop_xy[0]+crop_size[0]), int(crop_xy[1]+crop_size[1])))
  # resize it
	image = image.resize(dest_sz, Image.ANTIALIAS)
	return image



def rangeSensor():
	GPIO.setmode(GPIO.BCM)

	TRIG = 23 
	ECHO = 24

	print "Distance Measurement In Progress"

	GPIO.setup(TRIG,GPIO.OUT)
	GPIO.setup(ECHO,GPIO.IN)

	GPIO.output(TRIG, False)
	print "Waiting For Sensor To Settle"
	time.sleep(2)

	GPIO.output(TRIG, True)
	time.sleep(0.00001)
	GPIO.output(TRIG, False)

	while GPIO.input(ECHO)==0:
		pulse_start = time.time()

	while GPIO.input(ECHO)==1:
		pulse_end = time.time()

	pulse_duration = pulse_end - pulse_start

	distance = pulse_duration * 17150

	distance = round(distance, 2)

	print "Distance:",distance,"cm"

	GPIO.cleanup()
	
	return distance

	








if __name__ == '__main__':

    
	imagePath = "/home/pi/Project/faceRec/FACERECOGNITION_PROJECT/TestImg_3"
    #filename = "1.png"
	cascPath = "haarcascade_frontalface_default.xml"
	
	

	cam = VideoCapture(0)   # 0 -> index of camera

	while(1):
		
		distance = rangeSensor()
		
		if(distance<40.00):
	#--------------------------------------------------------
			#Take a picture from camera
			
			s, img = cam.read()
			if s:    # frame captured without any errors
				print "capture done!!"
				#cam.release()
			else:
				print "Not successful!!"
				

			
			#Convert to gray scale
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			
		#--------------------------------------------------------

			# read images
			[X,y] = read_images(imagePath)

			# Create the haar cascade
			faceCascade = cv2.CascadeClassifier(cascPath)

		 #--------------------------------------------------------
		  
			# compute the eigenfaces model
			#model = EigenfacesModel(X, y) #makes an instance of EigenfacesModel class
			
		#--------------------------------------------------------

			# For face recognition we will the the LBPH Face Recognizer 
			recognizer = cv2.createLBPHFaceRecognizer() #Better results in different lighting conditions
			
			#recognizer = cv2.createEigenFaceRecognizer()
			
			# Perform the tranining
			recognizer.train(X, np.array(y))
			
			# Detect faces in the image
			faces = faceCascade.detectMultiScale(
				gray,
				scaleFactor=1.1,
				minNeighbors=3,
				minSize=(100,100),
				flags =cv2.cv.CV_HAAR_SCALE_IMAGE
			)

			#Stores faces found
			faceArray = []

			if (len(faces) >0):
		  
				print "Found {0} faces!".format(len(faces))
				
				# Draw a rectangle around the faces
				for (x, y, w, h) in faces:
					cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)
					faceArray.append(gray[y: y + h, x: x + w])
					print [y+h, x+w]
				
				#cv2.imshow('Video', gray)
				
				for i in range(0,len(faceArray)):
		 
					im = np.asarray(faceArray[i], dtype=np.uint8)
					
					print faceArray[i].size
					
					im = cv2.resize(im, (92,112), interpolation = cv2.INTER_LINEAR)
					cv2.equalizeHist( im, im)
					
					imX = Image.fromarray(im)
					print imX.size
					imX.save("face" + str(i) + ".jpg")
					#imx = CropFace(imX, eye_left=(30,45), eye_right=(55,45), offset_pct=(0.3,0.3), dest_sz=(92,112))
					#imx.save("face.jpg")
					im = np.asarray(im)
					
					
					#cv2.equalizeHist( im, im)
					#[pre,con] = model.predict(im)
					
					[pre, dif] = recognizer.predict(im)
					

					print "predicted =", pre
					print "difference =", dif
					
					#if (dif<100.00):
					#	result = pre
						
					#else:
					#	result = common
					
					orig_string = str("/home/pi/Project/pythonWWW/noticeBoard/files/input.txt")
					f = open(orig_string,'r')
					temp = f.read()
					f.close()

					new_string = str("/home/pi/Project/pythonWWW/noticeBoard/files/input.txt")
					f = open(new_string, 'w')
					f.write("E" + str(pre) + '\n')

					f.write(temp)
					f.close()
			
				
				#cv2.waitKey(0) 
					
				
			else:
				print "No faces" 
				
			time.sleep(5)
		
	
    


#CropFace(image, eye_left=(300,700), eye_right=(700,700), offset_pct=(0.2,0.2), dest_sz=(92,112)).save("ss.jpg")


