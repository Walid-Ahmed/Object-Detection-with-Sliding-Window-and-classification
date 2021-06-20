'''
Credit

Adrian Rosebrock, urning any CNN image classifier into an object detector with Keras, TensorFlow, and OpenCV , PyImageSearch, https://www.pyimagesearch.com/2020/06/22/turning-any-cnn-image-classifier-into-an-object-detector-with-keras-tensorflow-and-opencv/, accessed on 16 April 2021


'''


# USAGE

# python detect_with_classifier_sliding_window.py --image images/test3.jpeg   --min-conf 0.8  --modelFile  soccer_Classifier.h5

# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
import imutils
import sys
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import imutils
import cv2
from tensorflow.keras.models import load_model
import os
import shutil
np.set_printoptions(threshold=sys.maxsize)





def sliding_window(image, step, ws):
	# slide a window across the image
	for y in range(0, image.shape[0] - ws[1], step):
		for x in range(0, image.shape[1] - ws[0], step):
			# yield the current window
			yield (x, y, image[y:y + ws[1], x:x + ws[0]])

def image_pyramid(image, scale=1.5, minSize=(224, 224)):
	# yield the original image
	yield image

	# keep looping over the image pyramid
	while True:
		# compute the dimensions of the next image in the pyramid
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)

		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break

		# yield the next image in the pyramid
		yield image







demoFrame = np.zeros([512,1300,3],dtype=np.uint8)
demoFrame.fill(255)


fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
fileNameToSaveVideo="demo_Object_Detection_with_Sliding_Window_and_binary_classification.mp4"
fps=70
video_creator = cv2.VideoWriter(fileNameToSaveVideo,fourcc, fps, (1300,512))


#  prepare folders to store  cropped rois  ,rois classiffied as positive  images  , rois classiffied as negative images for further analysis
foldersToCreate=["cropped" ,"positive","negative"]
for folderToCreate  in foldersToCreate:
	if os.path.exists(folderToCreate):
		shutil.rmtree(folderToCreate)
	os.mkdir(folderToCreate)	

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True,
	help="path to the input image")
ap.add_argument("-s", "--size", type=str, default="(200, 150)",
	help="ROI size (in pixels)")
ap.add_argument("--min-conf", type=float, default=0.9,
	help="minimum probability to filter weak detections")
ap.add_argument("--modelFile", type=str, default=-1,
	help="model used for classification")


args = vars(ap.parse_args())
args["modelFile"]
# initialize variables used for the object detection procedure
WIDTH = 600
PYR_SCALE = 1.5
WIN_STEP = 16
ROI_SIZE = (64, 64)
INPUT_SIZE = (32,32)     #This will be the input size to the model

# load our the network weights from disk
print("[INFO] loading network...")
pathToModelFile=args["modelFile"]
model = load_model(pathToModelFile)
print("[INFO] Model successfully loaded  from {}".format(pathToModelFile))


# load the input image from disk, resize it such that it has the
# has the supplied width, and then grab its dimensions
orig = cv2.imread(args["image"])
orig = imutils.resize(orig, width=WIDTH)
(H, W) = orig.shape[:2]

# initialize the image pyramid
pyramid = image_pyramid(orig, scale=PYR_SCALE, minSize=ROI_SIZE)


rois = [] # store rois
locs = [] #store coordinate of roi


i=949
# loop over the image pyramid
for image in pyramid:
	# determine the scale factor between the *original* image
	# dimensions and the *current* layer of the pyramid
	scale = W / float(image.shape[1])

	print("[INFO] Working with image with size  {}  after a scale factor {} ".format(image.shape,scale))


	# for each layer of the image pyramid, loop over the sliding
	# window locations

	#x,y are the coordinates   within image returned from pyramid
	for (x, y, roiOrig) in sliding_window(image, WIN_STEP, ROI_SIZE):


		# scale the (x, y)-coordinates of the ROI with respect to the
		# *original* image dimensions
		x_modified = int(x * scale)
		y_modified = int(y * scale)
		w_modified = int(ROI_SIZE[0] * scale)
		h_modified = int(ROI_SIZE[1] * scale)
		
		w= int(ROI_SIZE[0])
		h= int(ROI_SIZE[1])

		pathToSave=os.path.join("cropped","crop_"+str(i)+".png")
		cv2.imwrite(pathToSave, roiOrig) 
		i=i+1

		roi = cv2.resize(roiOrig, INPUT_SIZE)  #resize to input size needed for model
		roi = img_to_array(roi) #change image to numpy array
		roi=roi/255
	

		
		# update our list of ROIs and associated coordinates
		rois.append(roi)
		locs.append((x_modified, y_modified, x_modified + w_modified, y_modified + h_modified))



		clone=image.copy()
		cv2.rectangle(clone, (x, y), (x + w, y + h),(0, 255, 0), 2)


			
		h1,w1,_=clone.shape
		h2,w2,_=roiOrig.shape
		demoFrame.fill(255)
		a1=10+int(300-0.5*w1) +350
		b1=100
		b=300+350
		demoFrame[b1:b1+h1,a1:a1+w1]=clone
		demoFrame[10:10+h2,b:b+w2]=roiOrig
		cv2.imshow("Demo", demoFrame)
		video_creator.write(demoFrame)


		cv2.waitKey(1)



# convert the ROIs to a NumPy array
rois = np.array(rois, dtype="float32")

# classify each of the proposal ROIs using the model 
probs = model.predict(rois)

y_pred=probs.argmax(axis=1)



label="Scoccer Ball"
boxes=[]
probs_soccer=[]
# loop over the predictions

p=0
n=0
for (i, classIndex) in enumerate(y_pred):
	# grab the prediction information for the current ROI
	#(imagenetID, label, prob) = p[0]


	if classIndex== 1:    # the roi was classifies positive
		box = locs[i]
		out=probs[i]
		if (out[1]>= args["min_conf"]): ## filter out weak detections
			boxes.append(box)
			probs_soccer.append(out[1])
			positiveImage=rois[i]*255
			pathToSave=os.path.join("positive","_pos_"+str(p)+".png")
			p=p+1
			cv2.imwrite(pathToSave, positiveImage) 
		else:	
			pathToSave=os.path.join("negative","neg_"+str(n)+".png")
			n=n+1
			negativeImage=rois[i]*255
			cv2.imwrite(pathToSave, negativeImage) 

		





# loop over all bounding boxes for the current label
clone=orig.copy()
for (box, prob) in zip(boxes,probs_soccer):
	# draw the bounding box on the image
	(startX, startY, endX, endY) = box
	cv2.rectangle(clone, (startX, startY), (endX, endY),(0, 255, 0), 2)




h1,w1,_=clone.shape
demoFrame.fill(255)	
a=10+int(300-0.5*w1)
i=int((a+w1)/2)-220
j=30
cv2.putText(demoFrame, "Detected Soccer ball  before NMS", (i, j),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
demoFrame[100:100+h1,a:a+w1]=clone
video_creator.write(demoFrame)



boxes = np.array(boxes, dtype=np.float32)
boxes = non_max_suppression(boxes, probs_soccer)


clone = orig.copy()
for (startX, startY, endX, endY) in boxes:
		# draw the bounding box and label on the image
		cv2.rectangle(clone, (startX, startY), (endX, endY),
			(0, 255, 0), 2)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.putText(clone, label, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# show the output after apply non-maxima suppression
#cv2.imshow("After", clone)
i=int((a+w1)/2)-90+ w1-50

cv2.putText(demoFrame, "Detected Soccer ball after NMS", (i, j),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

demoFrame[100:100+h1,a+w1+20:a+w1+20+w1]=clone
cv2.imshow("Demo", demoFrame)

#shlow last frame for a longer time
for  i in range(300):
	video_creator.write(demoFrame)

cv2.waitKey(3000)
video_creator.release()

print("[INFO]   Demo video saved to file {}".format(fileNameToSaveVideo))



