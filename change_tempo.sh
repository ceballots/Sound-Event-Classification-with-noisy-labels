#!/usr/bin/bash

type_data=$1

if [ $# != 1 ]; then
  echo "Usage: change_tempo.sh  <type_data>"
  echo " e.g.: change_tempo.sh train"
  exit 1;
fi

cd ~/DataProcessed/processed_wavs_$type_data
#number_of_elements=${ls -1 | wc -l}

for element in $(ls);do
    echo changing speed of $element
    for speed in 0.7 0.8 0.9 1.2;do
       mkdir -p ../data_${type_data}_${speed}_speed
       sox -t wav $element -t wav ../data_${type_data}_${speed}_speed/$element-${speed}.wav tempo -s $speed
    done;
done
