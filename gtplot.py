import cv2
import os
import csv
from pathlib import Path

width = 3264     #X
height = 2448    #Y
color = (255,0,0)
thickness = 7

directory = "/home/ankit/Documents/ProductRecognition/datasets/GP-Test/images/"
directory_annot = '/home/ankit/Documents/ProductRecognition/datasets/GP-Test/labels/'
save_dir = "/home/ankit/Documents/ProductRecognition/CODE/yolov5/Grozi-3.2-Annot/Annotations(pseudo)"

# w=3264 h=2448

#files = Path(directory).glob('*.jpg')
files = Path(directory_annot).glob('*.csv')

for file in files:
    annot_path = os.path.join(directory_annot,file)
    pos_undr = annot_path.find('_')
    pos_dot = annot_path.find('.')
    file_num = annot_path[pos_undr + 1:pos_dot]
    #annot_path = "anno."+file_num + ".csv"
    image_path = file.name[:-3] + "jpg"
    print(image_path)
    #annot_path = os.path.join(directory_annot,annot_path)
    image_path_abs = os.path.join(directory,image_path)
    image = cv2.imread(image_path_abs)
    #image = cv2.resize(image,(width,height),interpolation = cv2.INTER_AREA)

    with open(annot_path) as csv_reader:
        annot_data = csv.reader(csv_reader,delimiter = ',')
        for row in annot_data:
            #print(row)
            #print(len(row))
            # x1 = float(row[0]) * width
            # x2 = float(row[1]) * width
            # y1 = float(row[2]) * height
            # y2 = float(row[3]) * height
            x1 = float(row[1]) #+ width/2
            y1 = float(row[2]) #+ height/2
            x2 = float(row[3]) #+ width/2
            y2 = float(row[4]) #+ height/2

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
			