import cv2
import os
import csv
from pathlib import Path

width = 900
height = 900
color = (0,0,255)
thickness = 2

directory = "/home/ankit/Documents/ProductRecognition/images[1]/"
directory_annot = '/home/ankit/Documents/ProductRecognition/CODE/yolov5/Grozi-3.2-Annot/Annotations_CSV/store1/BBox'
save_dir = "/home/ankit/Documents/ProductRecognition/CODE/yolov5/Grozi-3.2-Annot/Annotations_CSV/store1/annot_images"

files = Path(directory).glob('*.jpg')

for file in files:
    abs_path = os.path.join(directory,file)
    pos_undr = abs_path.find('_')
    pos_dot = abs_path.find('.')
    file_num = abs_path[pos_undr + 1:pos_dot]
    annot_path = "anno."+file_num + ".csv"
    annot_path = os.path.join(directory_annot,annot_path)

    image = cv2.imread(abs_path)
    image = cv2.resize(image,(width,height),interpolation = cv2.INTER_AREA)

    with open(annot_path) as csv_reader:
        annot_data = csv.reader(csv_reader,delimiter = ',')
        for row in annot_data:
            #print(row)
            #print(len(row))
            x1 = float(row[0]) * width
            x2 = float(row[1]) * width
            y1 = float(row[2]) * height
            y2 = float(row[3]) * height
            start_pt = (int(x1),int(y1))
            end_pt = (int(x2),int(y2))
            image = cv2.rectangle(image,start_pt,end_pt,color,thickness)
    
    annot_img_path = "anno."+file_num+".jpg"
    annot_img_path = os.path.join(save_dir,annot_img_path)
    cv2.imwrite(annot_img_path,image)
    # cv2.imshow("Test Image",image)
    # cv2.waitKey(700)
    # cv2.destroyAllWindows()
    
    

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
			