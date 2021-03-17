"""
Module to encode arXiv titles using the universal sentence encoder
Starting with the 'arxiv-cleaned.csv' file write the 'arxiv-encoded-full.csv' file


Functions:
----------
encodeTitle : Encode the 'title' column of a dataframe using the universal sentence encoder
"""

import numpy as np
import tensorflow_hub as hub
import csv


module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
embed = hub.load(module_url)

def encodeTitle(df):
    """
    Encode the 'title' column of a dataframe using the universal sentence encoder

    :param df: pd dataframe
    :return: pd daraframe

    """

    df["encoder"] = df["title"].map(lambda title: np.array(embed([title])[0]))

    for i in range(512):
        df["encoder" + str(i)] = df["encoder"].map(lambda encoder: encoder[i])

    return df

encoderColumns = []
for i in range(512):
    encoderColumns.append("encoding" + str(i))
categoryColumns = ["cs","physics","math"]


with open("arxiv-cleaned.csv" , "r") as cleaned:
    reader = csv.reader(cleaned)
    next(reader,None)
    with open("arxiv-encoded-full.csv", "w") as encoded:
        writer = csv.writer(encoded)
        writer.writerow(["title"] + encoderColumns + categoryColumns)
        count = 0
        for row in reader:
            title = row[0]
            categories = [row[1], row[2], row[3]]
            encoding = np.array(embed([title])[0]).tolist()
            toWrite = [title] + encoding + categories
            writer.writerow(toWrite)
            count += 1
            if count % 1000 == 0:
                print(count)
