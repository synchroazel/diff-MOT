import argparse
import os.path
import random
import shutil
import warnings

import cv2
import matplotlib
import pandas
from matplotlib import cm
from tqdm import tqdm

matplotlib.use('Agg')  # fixes matplotlib memory leak

warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)

FRAME_NAME_ZEROS = 6  # Meaning that frames names are zero padded to 6 digits


def main(args):
    tracker_dir = f"{args.trackers_folder}/{args.experiment_name}"
    output_folder = f"{args.output_folder}/{args.experiment_name}"

    # match args.experiment:
    #
    #     case 'MOT17_private':
    #         dataset_root = os.path.normpath(f"{args.datapath}/MOT17/train")
    #         video_folders = ['MOT17-02-DPM', ]
    #         # 'MOT17-02-FRCNN', 'MOT17-02-SDP',
    #         # 'MOT17-04-DPM', 'MOT17-04-FRCNN', 'MOT17-04-SDP',
    #         # 'MOT17-05-DPM', 'MOT17-05-FRCNN', 'MOT17-05-SDP',
    #         # 'MOT17-09-DPM', 'MOT17-09-FRCNN', 'MOT17-09-SDP',
    #         # 'MOT17-10-DPM', 'MOT17-10-FRCNN', 'MOT17-10-SDP',
    #         # 'MOT17-11-DPM', 'MOT17-11-FRCNN', 'MOT17-11-SDP',
    #         # 'MOT17-13-DPM', 'MOT17-13-FRCNN', 'MOT17-13-SDP', ]
    #         frames_folder_name = 'img1'
    #         ground_truth_folder_name = 'gt'
    #         output_folder = os.path.normpath(os.path.join(args.output_folder, "MOT17/train"))
    #
    #         ## !!!
    #
    #     case 'MOT20_private':
    #         dataset_root = os.path.normpath(f"{args.dataset_root}/MOT20/train")
    #         video_folders = ['MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05']
    #         frames_folder_name = 'img1'
    #         output_folder = os.path.normpath(os.path.join(args.output_folder, "MOT20/train"))
    #
    #     case 'MOT17_public':
    #         dataset_root = os.path.normpath(f"{args.dataset_root}/MOT17/train")
    #         video_folders = ['MOT17-02-DPM', 'MOT17-02-FRCNN', 'MOT17-02-SDP',
    #                          'MOT17-04-DPM', 'MOT17-04-FRCNN', 'MOT17-04-SDP',
    #                          'MOT17-05-DPM', 'MOT17-05-FRCNN', 'MOT17-05-SDP',
    #                          'MOT17-09-DPM', 'MOT17-09-FRCNN', 'MOT17-09-SDP',
    #                          'MOT17-10-DPM', 'MOT17-10-FRCNN', 'MOT17-10-SDP',
    #                          'MOT17-11-DPM', 'MOT17-11-FRCNN', 'MOT17-11-SDP',
    #                          'MOT17-13-DPM', 'MOT17-13-FRCNN', 'MOT17-13-SDP', ]
    #         frames_folder_name = 'img1'
    #         ground_truth_folder_name = 'gt'
    #         output_folder = os.path.normpath(os.path.join(args.output_folder, "MOT17/train"))
    #
    #     case 'MOT20_public':
    #         dataset_root = os.path.normpath(f"{args.dataset_root}/MOT20/train")
    #         video_folders = ['MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05']
    #         frames_folder_name = 'img1'
    #         output_folder = os.path.normpath(os.path.join(args.output_folder, "MOT20/train"))
    #
    #     case _:
    #         print("[ERR] Dataset not yet implemented")
    #         exit(1)

    if not os.path.exists(tracker_dir):
        print(f"[ERROR] Folder {tracker_dir} does not exist")
        exit(1)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Retrieve the name of the dataset used from the filename of one of the tracker files
    dataset_name = os.listdir(tracker_dir)[0].split('-')[0]

    for tracker in os.listdir(tracker_dir):

        track_name = tracker.split('.')[0]

        # Get frames list

        if track_name in os.listdir(os.path.normpath(os.path.join(args.datapath, dataset_name, "train"))):
            split = "train"
        elif track_name in os.listdir(os.path.normpath(os.path.join(args.datapath, dataset_name, "test"))):
            split = "test"
        else:
            print(f"[ERROR] Track {track_name} not found in dataset {dataset_name}")
            exit(1)

        frames_path = os.path.normpath(
            os.path.join(args.datapath, dataset_name, split, track_name))

        video_out_dir = os.path.normpath(os.path.join(output_folder, f'{track_name}_frames'))

        if not os.path.exists(video_out_dir):
            os.makedirs(video_out_dir)

        # get detections of test
        tracker_file_path = os.path.normpath(os.path.join(tracker_dir, tracker))
        detection_df = pandas.read_csv(tracker_file_path, sep=',',
                                       names=["frame",
                                              "id",
                                              "bb_left",
                                              "bb_top",
                                              "bb_width",
                                              "bb_height",
                                              "conf",
                                              "x",
                                              "y",
                                              "z"])

        # Drop useless columns (conf, x, y, z)
        detection_df = detection_df.iloc[:, 0:6]

        # Assign a color for each id in the frame
        ids = list(set(detection_df['id']))
        colormap = cm.get_cmap('tab20b', len(ids))  # colormap.colors to access them

        applied_colormap = dict()

        random.shuffle(ids)

        for i in range(len(ids)):
            applied_colormap[str(ids[i])] = colormap.colors[i]

        # Create each frame with bboxes
        frames = set(detection_df['frame'])

        # Initialize video
        # cpt = "private" if args.experiment.split("_")[-1] == "private" else "public"

        video_path = os.path.join(output_folder, f"{track_name}.mp4")

        # if not args.ffmpeg:
        #     video = cv2.VideoWriter(os.path.normpath(video_path), fourcc, 30.0, (1920, 1080))
        #
        # if args.ffmpeg:
        #     print("[INFO] User chose `--ffmpeg`, video will be created later from frames")

        for frame in tqdm(frames, desc=f"[TQDM] Processing frames from track {track_name}"):

            # Build frame name
            frame_name = str(frame)
            while len(frame_name) < FRAME_NAME_ZEROS:
                frame_name = '0' + frame_name
            frame_name += args.image_extension

            # get the info of that frame only
            frame_df = detection_df.loc[detection_df['frame'] == frame, :]

            # Load the image
            frame_path_in = os.path.normpath(os.path.join(frames_path, "img1", frame_name))
            frame_path_out = os.path.normpath(os.path.join(video_out_dir, frame_name))
            img = cv2.imread(frame_path_in)

            frame_ids = frame_df['id'].to_list()

            # Create bounding box for each id in the frame
            for id in frame_ids:
                color = tuple(applied_colormap[str(id)][0:3])
                color = tuple(255 * c for c in color)

                # Rectangle
                start_point = (int(frame_df.loc[frame_df['id'] == id, "bb_left"].values[0]),
                               int(frame_df.loc[frame_df['id'] == id, "bb_top"].values[0] +
                                   frame_df.loc[frame_df['id'] == id, "bb_height"].values[0]))  # bottom left corner
                end_point = (int(frame_df.loc[frame_df['id'] == id, "bb_left"].values[0] +
                                 frame_df.loc[frame_df['id'] == id, "bb_width"].values[0]),
                             int(frame_df.loc[frame_df['id'] == id, "bb_top"].values[0]))  # top right corner
                thicc = 3
                img = cv2.rectangle(img, start_point, end_point, color, thicc)

                # Text
                text_pos = (
                    int(frame_df.loc[frame_df['id'] == id, "bb_left"].values[0] +
                        frame_df.loc[frame_df['id'] == id, "bb_width"].values[0] / 5),
                    int(frame_df.loc[frame_df['id'] == id, "bb_top"].values[0] - 5)
                )
                img = cv2.putText(img, "obj " + str(id), text_pos, cv2.FONT_HERSHEY_SIMPLEX,  # PLAIN,
                                  0.8,  # font scale
                                  color, thicc - 2, cv2.LINE_AA)

                if img is None:
                    exit(1)

                cv2.imwrite(frame_path_out, img)

        print("[INFO] Creating video with ffmpeg")

        # Prepare a command for video creation with ffmpeg
        cmd = f"ffmpeg " \
              f"-framerate {args.fps} " \
              f"-pattern_type glob " \
              f"-i '{video_out_dir}/*{args.image_extension}' " \
              f"-c:a copy " \
              f"-shortest " \
              f"-c:v libx264 " \
              f"-pix_fmt yuv420p " \
              f"{video_path} " \
              f"-y"

        # Execute the command
        os.system(cmd)

        # Remove the frames folder and its content after creating the video
        shutil.rmtree(video_out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a video from MOT test detections')

    parser.add_argument('-d', '--datapath', type=str,
                        help='path to the dataset root folder')
    parser.add_argument('-e', '--experiment_name', type=str,
                        help='name of the experiment (folder containing the tracker file)')
    parser.add_argument('-t', '--trackers_folder', type=str, default='trackers',
                        help='path to the trackers folder')
    parser.add_argument('-o', '--output_folder', type=str, default='videos',
                        help='path to the output folder in which to save videos')
    parser.add_argument('-f', '--fps', type=str, default='30',
                        help='fps of the output video.')
    parser.add_argument('-i', '--image_extension', type=str, default='.jpg',
                        help='images extension')

    args = parser.parse_args()

    main(args)
