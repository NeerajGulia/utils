import cv2
import os
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video file")
ap.add_argument("-o", "--output", required=True,
	help="path to output directory where to save the extracted frames")
args = vars(ap.parse_args())


vidcap = cv2.VideoCapture(args["input"])
success, image = vidcap.read()
count = 0
directoryname = args["output"]
while success:
  cv2.imwrite(os.path.join(directoryname, "%d.jpg" % count), image)     # save frame as JPEG file      
  success,image = vidcap.read()
  count += 1
  if count % 50 == 0:
    print('saved {} frames: '.format(count))

print('total frames saved: ', count)    
