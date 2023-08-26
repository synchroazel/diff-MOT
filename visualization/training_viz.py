import argparse
import os.path

import matplotlib
import matplotlib.pyplot as plt

from utilities import *

matplotlib.use('Agg')  # fixes matplotlib memory leak

def plot_training(avg_train_losses, avg_val_losses, save_folder, exp_name, epoch):
    plt.figure(figsize=(10, 6))
    plt.plot(avg_train_losses, label="Training Loss")
    plt.plot(avg_val_losses, label="Validation Loss")
    plt.xlabel("Tracklets")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss vs Validation Loss, epoch " + str(epoch), fontsize=16, loc="left")
    plt.tight_layout()
    save_path = os.path.normpath(
        os.path.join(
            save_folder, exp_name, 'losses'
        )
    )
    create_folders(save_path)
    save_path = os.path.normpath(
        os.path.join(
            save_path,
            "training_vs_validation_loss_epoch_" + str(epoch) + ".png"
        )
    )
    plt.savefig(save_path)
    plt.close()


def plot_accuracy(avg_accuracy_on_0, avg_accuracy_on_1, save_folder, exp_name, epoch):
    plt.figure(figsize=(10, 6))
    plt.plot(avg_accuracy_on_0, label="Accuracy on 0")
    plt.plot(avg_accuracy_on_1, label="Accuracy on 1")
    plt.xlabel("Tracklets")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy on 0 vs Accuracy on 1", fontsize=16, loc="left")
    plt.ylim(0,1)
    plt.tight_layout()
    save_path = os.path.normpath(
        os.path.join(
            save_folder, exp_name, "accuracies"
        )
    )
    create_folders(save_path)
    save_path = os.path.normpath(
        os.path.join(
            save_path,
            "accuracy_comparison_epoch_" + str(epoch) + ".png"
        )
    )
    plt.savefig(save_path)
    plt.close()


def plot_error(avg_error_on_0, avg_error_on_1, save_folder, exp_name, epoch):
    plt.figure(figsize=(10, 6))
    plt.plot(avg_error_on_0, label="Error on 0")
    plt.plot(avg_error_on_1, label="Error on 1")
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.legend()
    plt.title("Error on 0 vs Error on 1", fontsize=16, loc="left")
    plt.ylim(0, 1)
    plt.tight_layout()
    save_path = os.path.normpath(
        os.path.join(
            save_folder, exp_name, 'errors'
        )
    )
    create_folders(save_path)
    save_path = os.path.normpath(
        os.path.join(
            save_path,
            "error_comparison_epoch_" + str(epoch) + ".png"
        )
    )
    plt.savefig(save_path)
    plt.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize training data from JSON.")

    parser.add_argument("-f", "--models_folder", type=str,
                        help="Path to the JSON files with training data.",
                        default="saves/models")
    parser.add_argument("-c", "--classification", action='store_true',
                        help="If true classification, otherwise regression")
    parser.add_argument("-m", "--model", type=str,
                        help="node model used",
                        default="timeaware")
    parser.add_argument("-p", "--predictor", type=str,
                        help="type of predictor",
                        default="node")
    parser.add_argument("-b", "--backbone", type=str,
                        help="backbone used",
                        default="resnet50")

    parser.add_argument("--layer_size", type=int,
                        help='layer_size',
                        default=500)
    parser.add_argument("--messages", type=int,
                        help="number of message passing steps",
                        default=6)
    parser.add_argument("--mot",
                        help="MOT of reference",
                        default='MOT17')


    parser.add_argument("-o", "--save_folder", type=str,
                        help="Path to the folder where plots will be saved.", default="images")

    args = parser.parse_args()

    models_path = args.models_folder
    task = 'classification' if args.classification else 'regression'
    model_name = args.model
    output_folder = args.save_folder

    predictor = args.predictor
    layer_size = str(args.layer_size)
    messages = str(args.messages)
    backbone = args.backbone
    mot =args.mot

    del args
    # setup folders
    common_path = os.path.normpath(
        os.path.join(
            task,
            model_name + "_base"
        )
    )
    output_folder = os.path.normpath(
        os.path.join(output_folder,
                     common_path
                     )
    )
    input_folder = os.path.normpath(
        os.path.join(models_path,
                     common_path
                     )
    )
    create_folders(output_folder)

    del models_path

    file_name = predictor + "-predictor_" + "node-model-" + model_name + \
        "_edge-model-base_layer-size-" + layer_size + "_backbone-" + backbone + \
        "_messages-" + messages+"_trained_on_"+ mot

    print("Producing images for: " + file_name)

    for epoch in range(0, 1000):

        file_path = os.path.normpath(
            os.path.join(
                input_folder,
                'Epoch_'+ str(epoch),
                file_name
            )
        )
        if not os.path.exists(file_path + ".json"):
            print("Epoch " + str(epoch) + " does not exist for this file.\nFinished")
            exit(0)
        else:
            print("Epoch " + str(epoch))

        with open(file_path+".json", 'r') as f:
            training_data = json.load(f)

            avg_accuracy_on_0 = training_data["avg_accuracy_on_0"]
            avg_accuracy_on_1 = training_data["avg_accuracy_on_1"]
            avg_error_on_0 = training_data["avg_error_on_0"]
            avg_error_on_1 = training_data["avg_error_on_1"]
            avg_train_losses = training_data["avg_train_losses"]
            avg_val_losses = training_data["avg_val_losses"]


        plot_training(avg_train_losses, avg_val_losses, output_folder, file_name, epoch)
        plot_accuracy(avg_accuracy_on_0, avg_accuracy_on_1, output_folder, file_name, epoch)
        plot_error(avg_error_on_0, avg_error_on_1, output_folder, file_name, epoch)

