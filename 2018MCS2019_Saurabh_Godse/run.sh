#!/bin/sh

 
if [ $1 = 1 ];
then
python Question1.py $2 $3 $4 $5        
elif [ $1 = 2 ];
then
python Question2.py $2 $3 $4 
elif [ $1 = 3 ];
then
python Question3.py $2 $3
else
python Question4.py $2 $3 $4
fi
#python Question4.py q4x.dat q4y.dat 1

