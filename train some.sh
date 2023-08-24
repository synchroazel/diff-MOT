#!/bin/bash


python train.py --MOTtrain "MOT20" --MOTvalidation 'MOT17' --epochs 5 -L focal -l 0.001 --dropout 0.2 --alpha 0.05 --gamma 3 --messages 8 --optimizer Adam
python train.py --MOTtrain "MOT20" --MOTvalidation 'MOT17' --epochs 5 -L focal -l 0.001 --dropout 0.2 --alpha 0.05 --gamma 3 --messages 8 --optimizer Rprop
python train.py --MOTtrain "MOT20" --MOTvalidation 'MOT17' --epochs 5 -L focal -l 0.001 --dropout 0.2 --alpha 0.05 --gamma 3 --messages 8 --optimizer SGD
python train.py --MOTtrain "MOT20" --MOTvalidation 'MOT17' --epochs 5 -L focal -l 0.001 --dropout 0.2 --alpha 0.05 --gamma 3 --messages 8 --optimizer RMSprop
python train.py --MOTtrain "MOT20" --MOTvalidation 'MOT17' --epochs 5 -L focal -l 0.001 --dropout 0.2 --alpha 0.05 --gamma 3 --messages 8 --optimizer Adagrad
python train.py --MOTtrain "MOT20" --MOTvalidation 'MOT17' --epochs 5 -L focal -l 0.001 --dropout 0.2 --alpha 0.05 --gamma 3 --messages 8 --optimizer Adamax
python train.py --MOTtrain "MOT20" --MOTvalidation 'MOT17' --epochs 5 -L focal -l 0.001 --dropout 0.2 --alpha 0.05 --gamma 3 --messages 8 --optimizer Adadelta