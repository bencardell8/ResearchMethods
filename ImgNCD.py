import os
import base64
import pandas as pd
import numpy as np
import zlib
import gzip
import bz2
from PIL import Image
import timeit
from sklearn.model_selection import train_test_split
from zipfile import ZipFile as zf


csv_data_1_file = "gen_1.csv"
csv_data_2_file = "gen_2.csv"
img_data_dir = "img_data/"
compressed_img_dir = "compressed_img/"
zip_type = "zlib"
k = 3


def read_data_from_csv(file_name: str, delimiter=',', columns_to_drop=None):
    data_frame = pd.read_csv(file_name)
    if columns_to_drop is not None:
        data_frame = data_frame.drop(columns=columns_to_drop)
    return data_frame


def split_dataframe_to_train_test(df: pd.DataFrame, col: str, test_frac=0.20, val_frac=0, random=None):
    """
    This function splits the pandas dataframe into train data and test data by default. Validation set can be
    obtained as well if specified.
    :param df: DataFrame to be split
    :param col: Column name for the data to be split
    :param test_frac: A floating point value between 0 and 1 to split the data as test data
    :param val_frac: A floating point value between 0 and 1 to split the data as validation data
    :param random: An integer to set the random seed or None
    :return:
    """
    x = df
    y = df[[col]]
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=test_frac, stratify=y, random_state=random)
    if val_frac == 0:
        return x_train, x_temp, None
    elif val_frac > 0:
        relative_frac_test = test_frac / (val_frac + test_frac)
        x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, stratify=y_temp, test_size=relative_frac_test,
                                                        random_state=random)
        return x_train, x_test, x_val
    else:
        raise ValueError("val_frac must be above 0")


def img_vectorise(name):
    img_name = ""
    if os.path.exists(img_data_dir + name + ".png"):
        img_name = name + ".png"
    elif os.path.exists(img_data_dir + name + ".jpg"):
        img_name = name + ".jpg"
    else:
        FileNotFoundError("No file with " + name + " found.")

    img = Image.open(img_data_dir+img_name)
    img_vector = np.asarray(img)
    print(f"Vector {img_vector}")
    vector_string = np.array2string(img_vector)

    return vector_string


def img_stringify(name):
    img_name = ""
    if os.path.exists(img_data_dir + name + ".png"):
        img_name = name+".png"
    elif os.path.exists(img_data_dir + name + ".jpg"):
        img_name = name+".jpg"
    else:
        FileNotFoundError("No file with " + name + " found.")

    with open(img_data_dir+img_name, "rb") as image2string:
        img_str = base64.b64encode(image2string.read())

    return img_str


def normalised_compression_distance(string_1, string_2, compressed_string_2_size):
    compressed_1_size = None
    compressed_1_2_size = None
    if zip_type == "zlib":
        compressed_1_size = len(zlib.compress(string_1))
        compressed_1_2_size = len(zlib.compress(string_1 + string_2))
    elif zip_type == "gzip":
        compressed_1_size = len(gzip.compress(string_1))
        compressed_1_2_size = len(gzip.compress(string_1 + string_2))
    elif zip_type == "bz2":
        compressed_1_size = len(bz2.compress(string_1))
        compressed_1_2_size = len(bz2.compress(string_1 + string_2))
    else:
        ValueError(f"{zip_type} not a valid format")

    ncd = (compressed_1_2_size - min(compressed_1_size, compressed_string_2_size)) / max(compressed_1_size, compressed_string_2_size)
    return ncd


def zip_files(files_to_zip):
    with zf('zipped_files.zip', 'w') as zipper:
        # writing each file one by one
        for file in files_to_zip:
            zipper.write(file)

# Get image -> Convert to vector -> Convert to String -> Compress/Ben's code ->


def main():
    data_1: pd.DataFrame = read_data_from_csv(csv_data_1_file)
    data_2: pd.DataFrame = read_data_from_csv(csv_data_2_file)
    data_1.drop(data_1[(data_1["Type1"] == "Dragon") | (data_1["Type1"] == "Fairy") | (data_1["Type1"] == "Ice") |
                       (data_1["Type1"] == "Ghost")].index, inplace=True)
    data_2.drop(data_2[(data_2["Type1"] == "Dragon") | (data_2["Type1"] == "Fairy") | (data_2["Type1"] == "Ice") |
                       (data_2["Type1"] == "Ghost")].index, inplace=True)
    data_1.reset_index(drop=True, inplace=True)
    data_2.reset_index(drop=True, inplace=True)
    ncds = pd.DataFrame(data=data_1["Type1"])
    # data_1["ImgStr"] = data_1["Name"].apply(img_stringify)
    # data_2["ImgStr"] = data_2["Name"].apply(img_stringify)
    data_1["ImgStr"] = data_1["Name"].apply(img_vectorise)
    data_2["ImgStr"] = data_2["Name"].apply(img_vectorise)
    print(data_1)
    # print(data_1)
    # unique_labels = pd.unique(data_1["Type1"])
    # group_dict = {}
    # for label in unique_labels:
    #     group_dict[label] = data_1.loc[data_1["Type1"] == label, "ImgStr"]
    #     # Shuffles the data
    #     # group_dict[label] = group_dict[label].sample(frac=1).reset_index(drop=True)

    for i in range(len(data_2)):
        if zip_type == "zlib":
            string = data_2.iloc[i, 2]
            compressed_string_size = len(zlib.compress(string))
            ncds["NCD"] = data_1["ImgStr"].apply(normalised_compression_distance, string_2=string, compressed_string_2_size=compressed_string_size)
            smallest_rows = ncds.nsmallest(k, "NCD", keep="all")
            # print(smallest_rows)


if __name__ == '__main__':
    start = timeit.default_timer()
    main()
    stop = timeit.default_timer()
    print('Time taken: ', stop - start)

