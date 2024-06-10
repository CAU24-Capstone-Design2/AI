#!/bin/bash
# Before execute whole process, training the wound segmentation model. Check wound_segmentation/train.ipynb file.
# start.sh $1(user_id) $2(filename and extension: NOT A ABSOLUTE PATH) "$3"(prompt) "$4"(keyword: tattoo style)
user=$1     
filename=$2 
prompt=$3   # Must be enclosed in quotation marks
keyword=$4  # Must be enclosed in quotation marks. Usable category: watercolor, neotraditional, realism

# wound segmentation 
cd ./wound_segmentation
conda run -n scart python main.py -u $user -f $filename 

# tattoo generation 
cd ../tattoo_generation
conda run -n scart python main.py -u $user -f $filename -p "$prompt" -l "$keyword"
