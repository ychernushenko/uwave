import os
import re

import numpy as np
import pandas as pd
import pyunpack
import requests


def download_data(
    remote_url: str,
    data_dir: str,
    output_file_name: str = "uWaveGestureLibrary.zip",
):
    """Method to download the Zip file from the given URL and save it to a specified directory"""

    os.makedirs(data_dir, exist_ok=True)
    data_file = os.path.join(data_dir, output_file_name)
    data = requests.get(remote_url)
    with open(data_file, "wb") as file:
        file.write(data.content)


def unzip_data(data_dir, input_file_name: str = "uWaveGestureLibrary.zip"):
    """Method to extract Zip file and rar files recursively"""

    arch = pyunpack.Archive(os.path.join(data_dir, input_file_name))
    arch.extractall(directory=data_dir)
    os.remove(os.path.join(data_dir, input_file_name))

    for root, _, file_names in os.walk(data_dir):
        for fn in file_names:
            if fn.endswith(".rar"):
                print("Extracting " + os.path.join(root, fn))
            else:
                print("Removing " + os.path.join(root, fn))
                os.remove(os.path.join(root, fn))
            if fn.endswith(".rar"):
                name = os.path.splitext(os.path.basename(fn))[0]
                try:
                    arch = pyunpack.Archive(os.path.join(root, fn))
                    item_dir = os.path.join(root, name)
                    os.mkdir(item_dir)
                    arch.extractall(directory=item_dir)
                    os.remove(os.path.join(root, fn))
                except Exception as e:
                    print("ERROR: BAD ARCHIVE " + os.path.join(root, fn))
                    print(e)


def extract_data(data_dir):
    """Method to extract data from text files and load them to a pandas DataFrame"""
    data_frames = []
    item_id = 1

    for root, _, file_names in os.walk(data_dir):
        for fn in file_names:
            if fn.endswith(".txt") and "Template_Acceleration" in fn:
                item_df = pd.read_table(
                    os.path.join(root, fn),
                    delimiter=" ",
                    header=None,
                    names=["x-acc", "y-acc", "z-acc", "gesture", "repetition"],
                )
                size = len(item_df)
                m = re.search("Template_Acceleration(.+?).txt", fn)
                ges_rep = m.group(1).split("-")
                gesture = int(ges_rep[0])
                if ges_rep[1] != "":
                    repetition = int(ges_rep[1])
                else:
                    repetition = np.nan
                item_df["gesture"] = [gesture] * size
                item_df["repetition"] = [repetition] * size
                item_df["item_id"] = [
                    item_id
                ] * size  # Create and populate item_id column
                data_frames.append(item_df)

                item_id += 1

    data_df = pd.concat(
        data_frames, ignore_index=True
    )  # Concatenate all item_df DataFrames
    return data_df


def load_uwave_dataset(
    remote_url: str = "http://zhen-wang.appspot.com/rice/files/uwave/uWaveGestureLibrary.zip",
    data_dir: str = "data",
):
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        download_data(remote_url, data_dir)
        unzip_data(data_dir)
    else:
        print("Data already exists. Skipping download and extraction.")
    data_df = extract_data(data_dir)
    X = np.array(data_df[["x-acc", "y-acc", "z-acc"]])
    y = np.array(data_df["gesture"])
    return X, y
