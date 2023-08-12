import argparse
import json
import os
import shutil

import matplotlib.pyplot as plt


def plot_training(avg_train_losses, avg_val_losses, save_folder, exp_name):
    plt.figure(figsize=(10, 6))
    plt.plot(avg_train_losses, label="Training Loss")
    plt.plot(avg_val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss vs Validation Loss", fontsize=16, loc="left")
    plt.tight_layout()
    plt.savefig(f"{save_folder}/{exp_name}/training_vs_validation_loss.png")
    plt.close()


def plot_accuracy(avg_accuracy_on_0, avg_accuracy_on_1, save_folder, exp_name):
    plt.figure(figsize=(10, 6))
    plt.plot(avg_accuracy_on_0, label="Accuracy on 0")
    plt.plot(avg_accuracy_on_1, label="Accuracy on 1")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy on 0 vs Accuracy on 1", fontsize=16, loc="left")
    plt.tight_layout()
    plt.savefig(f"{save_folder}/{exp_name}/accuracy_comparison.png")
    plt.close()


def plot_error(avg_error_on_0, avg_error_on_1, save_folder, exp_name):
    plt.figure(figsize=(10, 6))
    plt.plot(avg_error_on_0, label="Error on 0")
    plt.plot(avg_error_on_1, label="Error on 1")
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.legend()
    plt.title("Error on 0 vs Error on 1", fontsize=16, loc="left")
    plt.tight_layout()
    plt.savefig(f"{save_folder}/{exp_name}/error_comparison.png")
    plt.close()


def main(args):
    with open(args.json_path, 'r') as f:
        training_data = json.load(f)

        avg_accuracy_on_0 = training_data["avg_accuracy_on_0"]
        avg_accuracy_on_1 = training_data["avg_accuracy_on_1"]
        avg_error_on_0 = training_data["avg_error_on_0"]
        avg_error_on_1 = training_data["avg_error_on_1"]
        avg_train_losses = training_data["avg_train_losses"]
        avg_val_losses = training_data["avg_val_losses"]

        exp_name = args.json_path.split("/")[-1].split(".")[0]

    if not os.path.exists(f"{args.save_folder}/{exp_name}"):
        os.makedirs(f"{args.save_folder}/{exp_name}")

    plot_training(avg_train_losses, avg_val_losses, args.save_folder, exp_name)
    plot_accuracy(avg_accuracy_on_0, avg_accuracy_on_1, args.save_folder, exp_name)
    plot_error(avg_error_on_0, avg_error_on_1, args.save_folder, exp_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize training data from JSON.")

    parser.add_argument("-i", "--json-path", type=str,
                        help="Path to the JSON file with training data.")
    parser.add_argument("-o", "--save-folder", type=str,
                        help="Path to the folder where plots will be saved.")

    args = parser.parse_args()

    main(args)
