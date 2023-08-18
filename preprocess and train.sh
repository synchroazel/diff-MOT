#!/bin/bash

# python preprocess_data.py -m "MOT17" --classification --backbone 'efficientnet_v2_l'
# python preprocess_data.py -m "MOT17" --backbone 'efficientnet_v2_l'
# python preprocess_data.py -m "MOT17" --classification --backbone 'vit_l_32'
python preprocess_data.py -m "MOT17" --backbone 'vit_l_32'


python preprocess_data.py -m "MOT20" --classification --backbone 'efficientnet_v2_l'
python preprocess_data.py -m "MOT20" --backbone 'efficientnet_v2_l'
# python preprocess_data.py -m "MOT20" --classification --backbone 'vit_l_32'
# python preprocess_data.py -m "MOT20" --backbone 'vit_l_32'
















# python train.py -m "MOT17" --backbone 'resnet50' -L focal --epochs 25 --heads 1 --model 'timeaware' -l 0.001 --dropout 0.2 --classification
# python train.py -m "MOT17" --backbone 'resnet50' -L focal --epochs 25 --heads 1 --model 'timeaware' -l 0.001 --dropout 0.2
#
# python train.py -m "MOT17" --backbone 'resnet50' -L focal --epochs 25 --heads 6 --model 'timeaware' -l 0.001 --dropout 0.2 --classification --node_model
# python train.py -m "MOT17" --backbone 'resnet50' -L focal --epochs 25 --heads 6 --model 'timeaware' -l 0.001 --dropout 0.2 --node_model
#
# python train.py -m "MOT20" --backbone 'resnet50' -L focal --epochs 25 --heads 1 --model 'timeaware' -l 0.001 --dropout 0.2 --classification
# python train.py -m "MOT20" --backbone 'resnet50' -L focal --epochs 25 --heads 1 --model 'timeaware' -l 0.001 --dropout 0.2
#
# python train.py -m "MOT20" --backbone 'resnet50' -L focal --epochs 25 --heads 6 --model 'timeaware' -l 0.001 --dropout 0.2 --classification --node_model
# python train.py -m "MOT20" --backbone 'resnet50' -L focal --epochs 25 --heads 6 --model 'timeaware' -l 0.001 --dropout 0.2 --node_model



# python train.py -m "MOT17" --backbone 'vgg16' -L focal --epochs 25 --heads 1 --model 'timeaware' -l 0.001 --dropout 0.2 --classification
# python train.py -m "MOT17" --backbone 'vgg16' -L focal --epochs 25 --heads 1 --model 'timeaware' -l 0.001 --dropout 0.2
#
# python train.py -m "MOT17" --backbone 'vgg16' -L focal --epochs 25 --heads 6 --model 'timeaware' -l 0.001 --dropout 0.2 --classification --node_model
# python train.py -m "MOT17" --backbone 'vgg16' -L focal --epochs 25 --heads 6 --model 'timeaware' -l 0.001 --dropout 0.2 --node_model
#
# python train.py -m "MOT20" --backbone 'vgg16' -L focal --epochs 25 --heads 1 --model 'timeaware' -l 0.001 --dropout 0.2 --classification
# python train.py -m "MOT20" --backbone 'vgg16' -L focal --epochs 25 --heads 1 --model 'timeaware' -l 0.001 --dropout 0.2
#
# python train.py -m "MOT20" --backbone 'vgg16' -L focal --epochs 25 --heads 6 --model 'timeaware' -l 0.001 --dropout 0.2 --classification --node_model
# python train.py -m "MOT20" --backbone 'vgg16' -L focal --epochs 25 --heads 6 --model 'timeaware' -l 0.001 --dropout 0.2 --node_model