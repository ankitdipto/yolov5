This dataset is for academic research purposes only. It cannot be used commercially. All images are copyright to their respective owners.
---------------------------------------------------------------------------------------

This is the Grocery products dataset introduced in [1]. Please cite the following paper if you use the dataset in your work:

[1] Marian George, Christian Floerkemeier, "Recognizing Products: A Per-Exemplar Multi-Label Image Classification Approach", ECCV 2014.

If you have any questions or comments about the dataset, please send an e-mail to Marian George <marian.george@inf.ethz.ch>.

The dataset contains 8350 training images of grocery products, organized in 80 hierarchical classes, and 680 annotated test images of supermarket shelves. Additional 71 training images of shelves and price tags are in the 'Background' class.

****************************************************************************************

Contents:
---------

- TrainingFiles.txt: list of all training images of all classes.

- TestFiles.txt: list of all testing images.

- Training folder:
    - index.txt: number of training images in each class.
    
    - TrainingClassesIndex.mat: contains the names, and indices of all 81 classes.
    
    - Training images in JPG format, organized in hierarchical folders. Each specific product is represented by one image, taken in ideal studio conditions.
    
- Testing folder:
    - 680 annotated test images, where all the products are from the 27 training classes of the 'Food' category:
		1- Food/Bakery
		2- Food/Biscuits
		3- Food/Candy/Bonbons
		4- Food/Candy/Chocolate
		5- Food/Cereals
		6- Food/Chips
		7- Food/Coffee
		8- Food/Dairy/Cheese
		9- Food/Dairy/Creme
		10- Food/Dairy/Yoghurt
		11- Food/DriedFruitsAndNuts
		12- Food/Drinks/Choco
		13- Food/Drinks/IceTea
		14- Food/Drinks/Juices
		15- Food/Drinks/Milk
		16- Food/Drinks/SoftDrinks
		17- Food/Drinks/Water
		18- Food/Jars-Cans/Canned
		19- Food/Jars-Cans/Sauces
		20- Food/Jars-Cans/Spices
		21- Food/Jars-Cans/Spreads
		22- Food/Oil-Vinegar
		23- Food/Pasta
		24- Food/Rice
		25- Food/Snacks
		26- Food/Soups
		27- Food/Tea

** Typically, testing images taken in real stores do not cover all the products in the training images, as the training images represent catalog items that are not necessarily present in the stores.

    - Testing images are organized in 5 folders: store1-5,
    each folder has two subfolders:
        
        - images folder: the test images in JPG format, named %d.jpg.
        
        - annotation folder: MAT files named anno.%d.mat.
        
        Format of anno.%d.mat:
        
           annotation.class:
           -----------------
           a cell array of the class indices of the ground truth products. To get the names of the respective class indices, use the provided TrainingClassesIndex.mat file.
           
           annotation.label:
           -----------------
           a cell array of the specific product labels of the ground truth products.
           ** If a test image contains a product that is not present in the training images, it is labelled as '-1', or not labelled at all.
           
           annotation.bbox:
           ----------------
           relative coordinates of the bounding boxes of the ground truth products: [left_x, right_x, top_y, bottom_y]. 'x' coordinates are given relative to (ratio of) the image width. 'y' coordinates are given relative to the image height. 
           To get the absolute position of the coordinates, multiply each x value of the bounding box by the image width, and each y value by the image height. For example, if image height is h and width is w, and if the bounding box coordinates are [x1,x2,y1,y2], then the absolute coordinates of the upper left corner are: (x1*w, y1*h).
             	
             	
        
             	|----------------------------------> x
	       		|
		  		|
				|       
				|  (left_x, top_y)______
				|                |      | 
				|                |______| 
				|                       (right_x, bottom_y)
				|
				|
				|
				|
				v  y
           
