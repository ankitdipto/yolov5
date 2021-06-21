import os
import csv
from pathlib import Path

WIDTH = 3264     #X
HEIGHT = 2448    #Y
directory = "/home/ankit/Documents/ProductRecognition/datasets/GP-Test/labels"
save_dir = "/home/ankit/Documents/ProductRecognition/datasets/GP-Test/labels-txt"
files = Path(directory).glob('*.csv')

for file in files:
    abs_path = os.path.join(directory,file)
    save_path = file.name[:-3] + "txt"
    save_path = os.path.join(save_dir,save_path)
    with open(abs_path,'r') as csv_reader:
        annot_data = csv.reader(csv_reader,delimiter = ',')
        with open(save_path,"w") as txt_writer:
            for row in annot_data:
                row = list(map(int,row[1:]))
                Cls = 0
                x_centre = (row[0]+row[2])/2
                y_centre = (row[1] + row[3])/2
                box_wth = abs(row[2] - row[0])    # X_max - X_min
                box_ht = abs(row[3] - row[1])     # Y_max - Y_min

                x_centre, box_wth = x_centre/WIDTH ,box_wth/WIDTH
                y_centre, box_ht = y_centre/HEIGHT ,box_ht/HEIGHT

                entry = [Cls,x_centre,y_centre,box_wth,box_ht]
                entry = list(map(str,entry))
                txt_writer.write(" ".join(entry)+"\n")
