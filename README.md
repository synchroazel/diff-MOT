# Diffusion for Multi-Object Tracking

DiffMOT is a new (work in progress) approach to MOT which combines its traditional graph formulation with diffusion.

## Installation

1. Clone the repository (its `diffusion` branch!) with
```
git clone -b diffusion git@github.com:synchroazel/diff-MOT.git
```

2. Create a conda environment with the required packages
```
conda env create -f environment.yml
```

3. Activate the environment
```
conda activate diffmot
```

4. Place the MOT datasets in the `data` folder (see below). The folder structure should be as follows:
```
data
├── MOT17
│   ├── seqmaps
│   │   ├── MOT17-train-all.txt
│   │   ...
│   ├── train
│   │   ├── MOT17-02-DPM
│   │   │   ├── det
│   │   │   ├── gt
│   │   │   ├── img1
│   │   │   └── seqinfo.ini
│   │   ├── MOT17-02-FRCNN
│   │   ├── MOT17-02-SDP
│   │   ...
│   └── test
...
```

After making sure the folder structure looks okay, you can move to the next steps.

## Usage

### Preprocessing

To generate and store offline the graph features that will be used throughout the training, run
```
python preprocess_data.py --backbone <visual_backbone> --split <train/test> --detections <gt/det>
```

You can also use `-D` to change the default data folder (`data`) and `-m` to preprocess a specific MOT dataset (note
that the default is MOT17). Refer to the help function for more detailed information on the available arguments.

### Training

To train the model, run
```
python train.py [**cliargs]
```

Please refer to the help function for more detailed information on the available arguments.

### Evaluation

To evaluate the model, first you need to use a saved (.pkl) model to generate the predictions.
```
python produce_tracker.py --model <pkl_model> --experiment <exp_name_of_choice>
```

then you can evaluate the predictions using [**TrackEval**](https://github.com/JonathonLuiten/TrackEval] codebase), by
running
```
python eval.py --experiment <tracker> --seqmap <seqmap>
```

Running these snippets will use default names for output folders and assume the dasets were placed in `data`. It also
assumes the evaluation is being done on MOT17. You can play around with all these parameters and more, just use the
`--help` flag to see what's available for both functions.





