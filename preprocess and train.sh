# python preprocess_data.py -m "MOT20" --classification --backbone 'resnet50'
# python preprocess_data.py -m "MOT20" --backbone 'resnet50'
#
# python preprocess_data.py -m "MOT17" --classification --backbone 'resnet101'
# python preprocess_data.py -m "MOT17" --backbone 'resnet101'
# python preprocess_data.py -m "MOT20" --classification --backbone 'resnet101'
# python preprocess_data.py -m "MOT20" --backbone 'resnet101'
#
# python preprocess_data.py -m "MOT17" --classification --backbone 'vgg16'
# python preprocess_data.py -m "MOT17" --backbone 'vgg16'
# python preprocess_data.py -m "MOT20" --classification --backbone 'vgg16'
# python preprocess_data.py -m "MOT20" --backbone 'vgg16'
#
# python preprocess_data.py -m "MOT17" --classification --backbone 'vgg16'
# python preprocess_data.py -m "MOT17" --backbone 'vgg19'
# python preprocess_data.py -m "MOT20" --classification --backbone 'vgg16'
# python preprocess_data.py -m "MOT20" --backbone 'vgg19'



# python train.py -m "MOT17" --backbone 'resnet50' -L focal --epochs 25 --heads 1 --model 'timeaware' -l 0.001 --dropout 0.2 --classification
# python train.py -m "MOT17" --backbone 'resnet50' -L focal --epochs 25 --heads 1 --model 'timeaware' -l 0.001 --dropout 0.2

# python train.py -m "MOT17" --backbone 'resnet50' -L focal --epochs 25 --heads 6 --model 'timeaware' -l 0.001 --dropout 0.2 --classification --node_model
# python train.py -m "MOT17" --backbone 'resnet50' -L focal --epochs 25 --heads 6 --model 'timeaware' -l 0.001 --dropout 0.2 --node_model

# python train.py -m "MOT20" --backbone 'resnet50' -L focal --epochs 25 --heads 1 --model 'timeaware' -l 0.001 --dropout 0.2 --classification
# python train.py -m "MOT20" --backbone 'resnet50' -L focal --epochs 25 --heads 1 --model 'timeaware' -l 0.001 --dropout 0.2


# python train.py -m "MOT20" --backbone 'resnet50' -L focal --epochs 25 --heads 6 --model 'timeaware' -l 0.001 --dropout 0.2 --classification --node_model
# python train.py -m "MOT20" --backbone 'resnet50' -L focal --epochs 25 --heads 6 --model 'timeaware' -l 0.001 --dropout 0.2 --node_model



# python train.py -m "MOT17" --backbone 'resnet101' -L focal --epochs 25 --heads 1 --model 'timeaware' -l 0.001 --dropout 0.2 --classification
# python train.py -m "MOT17" --backbone 'resnet101' -L focal --epochs 25 --heads 1 --model 'timeaware' -l 0.001 --dropout 0.2

# python train.py -m "MOT17" --backbone 'resnet101' -L focal --epochs 25 --heads 6 --model 'timeaware' -l 0.001 --dropout 0.2 --classification --node_model
# python train.py -m "MOT17" --backbone 'resnet101' -L focal --epochs 25 --heads 6 --model 'timeaware' -l 0.001 --dropout 0.2 --node_model

# python train.py -m "MOT20" --backbone 'resnet101' -L focal --epochs 25 --heads 1 --model 'timeaware' -l 0.001 --dropout 0.2 --classification
# python train.py -m "MOT20" --backbone 'resnet101' -L focal --epochs 25 --heads 1 --model 'timeaware' -l 0.001 --dropout 0.2

# python train.py -m "MOT20" --backbone 'resnet101' -L focal --epochs 25 --heads 6 --model 'timeaware' -l 0.001 --dropout 0.2 --classification --node_model
# python train.py -m "MOT20" --backbone 'resnet101' -L focal --epochs 25 --heads 6 --model 'timeaware' -l 0.001 --dropout 0.2 --node_model




python train.py -m "MOT17" --backbone 'vgg16' -L focal --epochs 25 --heads 1 --model 'timeaware' -l 0.001 --dropout 0.2 --classification
python train.py -m "MOT17" --backbone 'vgg16' -L focal --epochs 25 --heads 1 --model 'timeaware' -l 0.001 --dropout 0.2

python train.py -m "MOT17" --backbone 'vgg16' -L focal --epochs 25 --heads 6 --model 'timeaware' -l 0.001 --dropout 0.2 --classification --node_model
python train.py -m "MOT17" --backbone 'vgg16' -L focal --epochs 25 --heads 6 --model 'timeaware' -l 0.001 --dropout 0.2 --node_model

python train.py -m "MOT20" --backbone 'vgg16' -L focal --epochs 25 --heads 1 --model 'timeaware' -l 0.001 --dropout 0.2 --classification
python train.py -m "MOT20" --backbone 'vgg16' -L focal --epochs 25 --heads 1 --model 'timeaware' -l 0.001 --dropout 0.2

python train.py -m "MOT20" --backbone 'vgg16' -L focal --epochs 25 --heads 6 --model 'timeaware' -l 0.001 --dropout 0.2 --classification --node_model
python train.py -m "MOT20" --backbone 'vgg16' -L focal --epochs 25 --heads 6 --model 'timeaware' -l 0.001 --dropout 0.2 --node_model