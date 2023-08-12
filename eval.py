from TrackEval.scripts.run_mot_challenge import evaluate_mot17
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--tracker_path', type=str, default='trackers',
                        help='Path to the folder containing all trackers.')

    parser.add_argument('--split', type=str, default='MOT17-train-all',
                        help='The name of the seqmap containing the list of tracks to evaluate.\n'
                             'All seqmaps for MOTX are assumed to be in MOTX/seqmaps.')

    parser.add_argument('--data_path', type=str, default='/media/dmmp/vid+backup/Data',
                        help='Path to the folder containing all datasets.')

    parser.add_argument('--output_path', type=str, default='output',
                        help='Path to the folder where the results will be saved.')

    # Not really sure of its role
    parser.add_argument('--tracker_sub_folder', type=str, default='exp_classification',
                        help='Subfolder of tracker_path where the tracker files are located.')

    args = parser.parse_args()

    evaluate_mot17(tracker_path=args.tracker_path,
                   split=args.split,
                   data_path=args.data_path,
                   tracker_sub_folder=args.tracker_sub_folder,
                   output_sub_folder=args.output_path)

    # Assume you have something like this in the root of the project:
    # trackers/
    #   exp1/
    #     MOT17-02-DPM.txt

    # Esempio funzionante:
    #
    # evaluate_mot17(tracker_path="trackers",
    #                split="MOT17-prova",
    #                data_path="data",
    #                tracker_sub_folder="exp1",
    #                output_sub_folder="boh2")
