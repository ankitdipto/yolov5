import cv2
import os
from pathlib import Path

directory = "/home/ankit/Documents/ProductRecognition/images[1]/"
files = Path(directory).glob('*.jpg')

for file in files:
    abs_path = os.path.join(directory,file)
    print(abs_path)
    image = cv2.imread(abs_path)
    image = cv2.resize(image,(900,900),interpolation = cv2.INTER_AREA)
    cv2.imshow("Test Image",image)
    cv2.waitKey(100)
    cv2.destroyAllWindows()
    
    

# x1 = 900 * 0.0027574
# y1 = 900 * 0.011846

# x2 = 900 * 0.36979
# y2 = 900 * 0.26430

# start_point = (int(x1),int(y1)) 
# end_point = (int(x2), int(y2)) 
 
# # Blue color in BGR 
# color = (100, 56, 218) 
  
# # Line thickness of 2 px 
# thickness = 2
  
# # Using cv2.rectangle() method 
# # Draw a rectangle with blue line borders of thickness of 2 px 
# image = cv2.rectangle(image, start_point, end_point, color, thickness)
# cv2.imshow("ground truths",image)
# cv2.waitKey(0)
# cv2.destroyAllWindows() 
			