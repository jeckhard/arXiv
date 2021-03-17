"""
Module to load in the arXiv metadata and clean it.
Operations carried out:
- load in the 'id', 'title' and 'categories' attributes from 'arxiv-metadata.json'
- clean the categories to only contain 'cs', 'physics', 'math' and 'other'
- add a label for each category and encode it into a boolean, 'other' is dropped
- clean the title by removing new lines
- write the data to 'arxiv-cleaned.csv'

Functions:
----------
loadLinesjson : Loads in part of a .json file
cleanCategories : Cleans arXiv categories to 'cs', 'physics', 'math' and 'other'
addCats : Add category labels to a data set
cleanTitles : Remove new lines from a title
"""


import json
import pandas as pd

def loadLinesjson(fileName,keys=[],numLines=None):
    """
    Loads in part of a .json file

    :param fileName: str
        path to the file
    :param keys: list of str elements
        list of attributes to be loaded in
    :param numLines: int or None
        number of lines to be loaded in
        default=None, representing all lines
    :return: pd dataframe
        loaded in data
    """

    loaded = []

    if not numLines:
        numLines = sum(1 for _ in open(fileName))

    with open(fileName, 'r') as file:
        for i in range(numLines):
            fullData = json.loads(file.readline())
            selectedData = {}
            for key in keys:
                if key in fullData:
                    selectedData[key] = fullData[key]
                else:
                    selectedData[key] = None
            loaded.append(selectedData)
        data = pd.DataFrame(loaded)

    return data


def cleanCategories(categories):
    """
    Cleans arXiv categories to 'cs', 'physics', 'math' and 'other'

    :param categories: str
        string containing all raw categories
    :return: set of str elements
        cleaned categories
    """

    categoryList = categories.split(" ")
    physics = {"astro-ph", "cond-mat", "gr-qc", "hep-ex", "hep-lat", "hep-ph", "hep-th", "math-ph", "nlin", "nucl-ex","nucl-th", "physics", "quant-ph"}
    labels = {"cs", "math", "physics"}


    categorySet=set()

    for category in categoryList:
        if "." in category:
            index = category.index(".")
            category = category[:index]
        if category in physics:
            categorySet.add("physics")
        elif category in labels:
            categorySet.add(category)
        else:
            categorySet.add("other")

    return categorySet


def addCats(data,labels):
    """
    Add category labels to a data set

    :param data: pd dataframe
    :param labels: list of str elements
        list of labels to be added
    :return: pd dataframe
    """
    for label in labels:
        data[label] = data["categories"].apply(lambda cat: int(label in cat))

    return data


def cleanTitles(title):
    """
    Remove new lines from a title

    :param title: str
    :return: str
    """
    title = title.replace("\n", " ")

    return title


data = loadLinesjson("arxiv-metadata.json",keys=["id","title","categories"])
data["categories"] = data["categories"].apply(cleanCategories)
data = addCats(data,{"cs", "math", "physics", "other"})
data = data.drop(columns=["id","categories","other"])
data["title"] = data["title"].apply(cleanTitles)
data.to_csv("arxiv-cleaned.csv",index=False)
