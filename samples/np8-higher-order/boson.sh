#!/bin/sh
#BSUB -n 1
#BSUB -o boson
#BSUB -J boson

date
python generate.py B     10.0 >> output-boson
date
python generate.py B    100.0 >> output-boson
date
python generate.py B   1000.0 >> output-boson
date
python generate.py B  10000.0 >> output-boson
date
