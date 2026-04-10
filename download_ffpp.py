#!/usr/bin/env python
"""Downloads FaceForensics++ and Deep Fake Detection public data release."""
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
import sys
import tempfile
import time
import urllib.request
from os.path import join

from tqdm import tqdm


FILELIST_URL = "misc/filelist.json"
DEEPFEAKES_DETECTION_URL = "misc/deepfake_detection_filenames.json"
DEEPFAKES_MODEL_NAMES = ["decoder_A.h5", "decoder_B.h5", "encoder.h5"]

DATASETS = {
    "original_youtube_videos": "misc/downloaded_youtube_videos.zip",
    "original_youtube_videos_info": "misc/downloaded_youtube_videos_info.zip",
    "original": "original_sequences/youtube",
    "DeepFakeDetection_original": "original_sequences/actors",
    "Deepfakes": "manipulated_sequences/Deepfakes",
    "DeepFakeDetection": "manipulated_sequences/DeepFakeDetection",
    "Face2Face": "manipulated_sequences/Face2Face",
    "FaceShifter": "manipulated_sequences/FaceShifter",
    "FaceSwap": "manipulated_sequences/FaceSwap",
    "NeuralTextures": "manipulated_sequences/NeuralTextures",
}
ALL_DATASETS = [
    "original",
    "DeepFakeDetection_original",
    "Deepfakes",
    "DeepFakeDetection",
    "Face2Face",
    "FaceShifter",
    "FaceSwap",
    "NeuralTextures",
]
COMPRESSION = ["raw", "c23", "c40"]
TYPE = ["videos", "masks", "models"]
SERVERS = ["EU", "EU2", "CA"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Downloads FaceForensics v2 public data release.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("output_path", type=str, help="Output directory.")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="all",
        choices=list(DATASETS.keys()) + ["all"],
        help="Which dataset to download.",
    )
    parser.add_argument(
        "-c",
        "--compression",
        type=str,
        default="raw",
        choices=COMPRESSION,
        help="Compression degree.",
    )
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        default="videos",
        choices=TYPE,
        help="Which file type to download.",
    )
    parser.add_argument(
        "-n",
        "--num_videos",
        type=int,
        default=None,
        help="Limit the number of videos to download.",
    )
    parser.add_argument(
        "--server",
        type=str,
        default="EU2",
        choices=SERVERS,
        help="Server to download the data from.",
    )
    args = parser.parse_args()

    if args.server == "EU":
        server_url = "http://canis.vc.in.tum.de:8100/"
    elif args.server == "EU2":
        server_url = "http://kaldir.vc.in.tum.de/faceforensics/"
    elif args.server == "CA":
        server_url = "http://falas.cmpt.sfu.ca:8100/"
    else:
        raise ValueError(f"Wrong server name. Choices: {SERVERS}")

    args.tos_url = server_url + "webpage/FaceForensics_TOS.pdf"
    args.base_url = server_url + "v3/"
    args.deepfakes_model_url = (
        server_url + "v3/manipulated_sequences/Deepfakes/models/"
    )
    return args


def download_files(filenames, base_url, output_path, report_progress=True):
    os.makedirs(output_path, exist_ok=True)
    iterable = tqdm(filenames) if report_progress else filenames
    for filename in iterable:
        download_file(base_url + filename, join(output_path, filename))


def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration)) if duration > 0 else 0
    percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
    sys.stdout.write(
        "\rProgress: %d%%, %d MB, %d KB/s, %d seconds passed"
        % (percent, progress_size / (1024 * 1024), speed, duration)
    )
    sys.stdout.flush()


def download_file(url, out_file, report_progress=False):
    out_dir = os.path.dirname(out_file)
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.isfile(out_file):
        fh, out_file_tmp = tempfile.mkstemp(dir=out_dir)
        os.close(fh)
        if report_progress:
            urllib.request.urlretrieve(url, out_file_tmp, reporthook=reporthook)
        else:
            urllib.request.urlretrieve(url, out_file_tmp)
        os.replace(out_file_tmp, out_file)
    else:
        tqdm.write(f"WARNING: skipping download of existing file {out_file}")


def main(args):
    print(
        "By pressing any key to continue you confirm that you have agreed to the "
        "FaceForensics terms of use as described at:"
    )
    print(args.tos_url)
    print("***")
    print("Press any key to continue, or CTRL-C to exit.")
    _ = input("")

    datasets = [args.dataset] if args.dataset != "all" else ALL_DATASETS
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    for dataset in datasets:
        dataset_path = DATASETS[dataset]

        if "original_youtube_videos" in dataset:
            print("Downloading original youtube videos.")
            suffix = "" if "info" not in dataset_path else "info"
            if not suffix:
                print("Please be patient, this may take a while (~40gb)")
            download_file(
                args.base_url + "/" + dataset_path,
                out_file=join(output_path, f"downloaded_videos{suffix}.zip"),
                report_progress=True,
            )
            return

        print(f'Downloading {args.type} of dataset "{dataset_path}"')

        if "DeepFakeDetection" in dataset_path or "actors" in dataset_path:
            filepaths = json.loads(
                urllib.request.urlopen(
                    args.base_url + "/" + DEEPFEAKES_DETECTION_URL
                ).read().decode("utf-8")
            )
            filelist = (
                filepaths["actors"]
                if "actors" in dataset_path
                else filepaths["DeepFakesDetection"]
            )
        elif "original" in dataset_path:
            file_pairs = json.loads(
                urllib.request.urlopen(args.base_url + "/" + FILELIST_URL)
                .read()
                .decode("utf-8")
            )
            filelist = []
            for pair in file_pairs:
                filelist += pair
        else:
            file_pairs = json.loads(
                urllib.request.urlopen(args.base_url + "/" + FILELIST_URL)
                .read()
                .decode("utf-8")
            )
            filelist = []
            for pair in file_pairs:
                filelist.append("_".join(pair))
                if args.type != "models":
                    filelist.append("_".join(pair[::-1]))

        if args.num_videos is not None and args.num_videos > 0:
            print(f"Downloading the first {args.num_videos} videos")
            filelist = filelist[: args.num_videos]

        dataset_videos_url = (
            args.base_url + f"{dataset_path}/{args.compression}/{args.type}/"
        )
        dataset_mask_url = args.base_url + f"{dataset_path}/masks/videos/"

        if args.type == "videos":
            dataset_output_path = join(output_path, dataset_path, args.compression, args.type)
            print(f"Output path: {dataset_output_path}")
            filelist = [filename + ".mp4" for filename in filelist]
            download_files(filelist, dataset_videos_url, dataset_output_path)
        elif args.type == "masks":
            dataset_output_path = join(output_path, dataset_path, args.type, "videos")
            print(f"Output path: {dataset_output_path}")
            if "original" in dataset:
                if args.dataset != "all":
                    print("Only videos available for original data. Aborting.")
                    return
                print("Only videos available for original data. Skipping original.\n")
                continue
            if "FaceShifter" in dataset:
                print("Masks not available for FaceShifter. Aborting.")
                return
            filelist = [filename + ".mp4" for filename in filelist]
            download_files(filelist, dataset_mask_url, dataset_output_path)
        else:
            if dataset != "Deepfakes":
                print("Models only available for Deepfakes. Aborting")
                return
            dataset_output_path = join(output_path, dataset_path, args.type)
            print(f"Output path: {dataset_output_path}")
            for folder in tqdm(filelist):
                folder_base_url = args.deepfakes_model_url + folder + "/"
                folder_dataset_output_path = join(dataset_output_path, folder)
                download_files(
                    DEEPFAKES_MODEL_NAMES,
                    folder_base_url,
                    folder_dataset_output_path,
                    report_progress=False,
                )


if __name__ == "__main__":
    main(parse_args())
