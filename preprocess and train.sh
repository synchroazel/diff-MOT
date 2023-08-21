#!/bin/bash


#
# python train.py -m "MOT17" --backbone 'resnet50' -L focal --epochs 25 --heads 6 --model 'timeaware' -l 0.001 --dropout 0.2 --node_model
# python train.py -m "MOT17" --backbone 'resnet50' -L focal --epochs 25 --heads 6 --model 'timeaware' -l 0.001 --dropout 0.2 --classification
#
# python train.py -m "MOT17" --backbone 'vit_l_32' -L focal --epochs 25 --heads 6 --model 'timeaware' -l 0.001 --dropout 0.2 --node_model
# python train.py -m "MOT17" --backbone 'vit_l_32' -L focal --epochs 25 --heads 6 --model 'timeaware' -l 0.001 --dropout 0.2 --classification

# python train.py -m "MOT17" --backbone 'efficientnet_v2_l' -L focal --epochs 25 --heads 6 --model 'timeaware' -l 0.001 --dropout 0.2 --node_model
# python train.py -m "MOT17" --backbone 'efficientnet_v2_l' -L focal --epochs 25 --heads 6 --model 'timeaware' -l 0.001 --dropout 0.2 --classification
#
#
#
# python train.py -m "MOT20" --backbone 'resnet50' -L focal --epochs 25 --heads 6 --model 'timeaware' -l 0.001 --dropout 0.2 --node_model
# python train.py -m "MOT20" --backbone 'resnet50' -L focal --epochs 25 --heads 6 --model 'timeaware' -l 0.001 --dropout 0.2 --classification
#
# python train.py -m "MOT20" --backbone 'vit_l_32' -L focal --epochs 25 --heads 6 --model 'timeaware' -l 0.001 --dropout 0.2 --node_model
# python train.py -m "MOT20" --backbone 'vit_l_32' -L focal --epochs 25 --heads 6 --model 'timeaware' -l 0.001 --dropout 0.2 --classification

# python train.py -m "MOT20" -M 'MOT20' --backbone 'efficientnet_v2_l' -L focal --epochs 25 --heads 6 --model 'timeaware' -l 0.001 --dropout 0.2 --node_model
# python train.py -m "MOT20" --backbone 'efficientnet_v2_l' -L focal --epochs 25 --heads 6 --model 'timeaware' -l 0.001 --dropout 0.2 --classification





# python train.py -m "MOT17" --backbone 'vgg16' -L focal --epochs 25 --heads 1 --model 'timeaware' -l 0.001 --dropout 0.2 --classification
# python train.py -m "MOT17" --backbone 'vgg16' -L focal --epochs 25 --heads 6 --model 'timeaware' -l 0.001 --dropout 0.2 --node_model
#
# python train.py -m "MOT20" --backbone 'vgg16' -L focal --epochs 25 --heads 1 --model 'timeaware' -l 0.001 --dropout 0.2 --classification
# python train.py -m "MOT20" --backbone 'vgg16' -L focal --epochs 25 --heads 6 --model 'timeaware' -l 0.001 --dropout 0.2 --node_model


# python train.py -m "MOT17" -M 'MOT17' -L focal --epochs 25 --heads 6 --model 'transformer' -l 0.001 --dropout 0.2 --node_model
# python train.py -m "MOT20" -M 'MOT20' -L focal --epochs 25 --heads 6 --model 'transformer' -l 0.001 --dropout 0.2 --classification
#
# python train.py -m "MOT17" -M 'MOT17' -L focal --epochs 25 --heads 6 --model 'attention' -l 0.001 --dropout 0.2 --node_model
# python train.py -m "MOT20" -M 'MOT20' -L focal --epochs 25 --heads 6 --model 'attention' -l 0.001 --dropout 0.2 --classification