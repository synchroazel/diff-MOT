import argparse
import os.path

import matplotlib
import matplotlib.pyplot as plt

from utilities import *

matplotlib.use('Agg')  # fixes matplotlib memory leak

plt.rc('font', family='serif')


def plot_training(avg_train_losses, epoch, save_folder, exp_name):
    plt.figure(figsize=(7, 5))
    plt.plot(avg_train_losses, label="Training Loss")
    plt.xlabel("Tracklets", fontsize=12)
    plt.xticks(fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.title(f"Training Loss over tracklets at epoch #{epoch}", fontsize=16, loc="left")
    plt.tight_layout()
    save_path = os.path.normpath(
        os.path.join(
            save_folder, exp_name
        )
    )
    create_folders(save_path)
    save_path = os.path.normpath(
        os.path.join(
            save_path,
            f"training_loss_{exp_name}_ep{epoch}.png"
        )
    )
    plt.savefig(save_path, dpi=400)
    plt.close()


def plot_validation(avg_accuracy_on_0, avg_accuracy_on_1, avg_error_on_0, avg_error_on_1, epochs, save_folder,
                    exp_name):
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, avg_accuracy_on_0, label="Accuracy on 0s")
    plt.scatter(epochs, avg_accuracy_on_0, s=10)
    plt.plot(epochs, avg_accuracy_on_1, label="Accuracy on 1s")
    plt.scatter(epochs, avg_accuracy_on_1, s=10)
    plt.plot(epochs, avg_error_on_0, label="Error on 0s")
    plt.scatter(epochs, avg_error_on_0, s=10)
    plt.plot(epochs, avg_error_on_1, label="Error on 1s")
    plt.scatter(epochs, avg_error_on_1, s=10)
    plt.xticks(epochs)
    plt.xticks(fontsize=12)
    plt.xlabel("Epochs", fontsize=12)
    plt.xticks(fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.legend(fontsize=12)
    plt.title("Validation accuracy during training", fontsize=16, loc="left")
    # plt.ylim(0, 1)
    save_path = os.path.normpath(
        os.path.join(
            save_folder, exp_name
        )
    )
    create_folders(save_path)
    save_path = os.path.normpath(
        os.path.join(
            save_path,
            f"validation_accuracy_{exp_name}.png"
        )
    )
    plt.savefig(save_path, dpi=400)
    plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Visualize training data from JSON.")

    parser.add_argument("-f", "--input_folder", type=str,
                        help="Path to the folder containing training data for each epoch."
                             "The folder must contain a series of .json files ending with '_epX', with X the epoch no.")

    parser.add_argument("-o", "--save_folder", type=str,
                        help="Path to the folder where plots will be saved.", default="images")

    parser.add_argument("-e", "--exp_name", type=str,
                        help="Name of the experiment.", default="exp")

    args = parser.parse_args()

    json_files = [file for file in os.listdir(args.input_folder) if file.endswith(".json")]

    avg_accuracy_on_0, avg_accuracy_on_1, avg_error_on_0, avg_error_on_1, avg_val_losses = [], [], [], [], []
    epochs = []

    for file in sorted(json_files):
        try:
            epoch = int(file.split("_ep")[-1].split(".")[0])
        except ValueError:
            "The input folder must contain a series of .json files ending with '_epX', with X the epoch number."

        with open(os.path.join(args.input_folder, file), 'r') as f:
            epoch_info = json.load(f)

            avg_accuracy_on_0 += epoch_info["avg_accuracy_on_0"]
            avg_accuracy_on_1 += epoch_info["avg_accuracy_on_1"]
            avg_error_on_0 += epoch_info["avg_error_on_0"]
            avg_error_on_1 += epoch_info["avg_error_on_1"]

            avg_train_losses = epoch_info["avg_train_losses"]

            epochs.append(epoch)

            plot_training(avg_train_losses, epoch, args.save_folder, args.exp_name)

    plot_validation(avg_accuracy_on_0, avg_accuracy_on_1,
                    avg_error_on_0, avg_error_on_1,
                    epochs, args.save_folder, args.exp_name)
