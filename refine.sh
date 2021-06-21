#!/bin/bash

dir='/home/ankit/Documents/ProductRecognition/datasets/GP-Test'
file='/home/ankit/Documents/ProductRecognition/datasets/GP-Test/temp.txt'  
i=1  
while read line; do  
    cp $dir/'images'/$line $dir/'IMAGES'/
    i=$((i+1)) 
done < $file  