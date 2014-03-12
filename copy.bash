#!/bin/bash

for i in `ls -1 ./`
do
    if [ -f ../$i ];
    then
        cp ../$i $i
    fi
done
