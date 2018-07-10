#!/bin/sh
#BSUB -n 1
#BSUB -o fermion
#BSUB -J fermion

date
python generate.py F     10.0 >> output-fermion
date
python generate.py F    100.0 >> output-fermion
date
python generate.py F   1000.0 >> output-fermion
date
python generate.py F  10000.0 >> output-fermion
date
