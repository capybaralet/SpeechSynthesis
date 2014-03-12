#!/bin/bash

for i in `ls -1 ./`
do
    if [ -f $i ];
    then
        printf "\n\n         FILENAME: "
        echo $i 
        diff ../$i $i
    fi
done
