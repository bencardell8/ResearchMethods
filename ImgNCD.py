import os
import pandas as pd
import numpy as np
import zlib
import gzip
import bz2
from PIL import Image
import timeit
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


csv_full_data_file = "pokemon.csv"
csv_data_1_file = "gen_1.csv"
csv_data_2_file = "gen_2.csv"
img_data_dir = "img_data/"
compressed_img_dir = "compressed_img/"
zip_type = "zlib"  # zlib, gzip, bz2
# k = 11


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
    # norm_img = img.convert("P", palette=Image.ADAPTIVE, colors=8)
    # img_vector = np.asarray(norm_img)
    image_1d = img_vector.reshape(-1)
    # print(f"Vector {img_vector}")
    # vector_string = np.array2string(img_vector)
    vector_string = ''.join([str(num) for num in image_1d])
    # print(vector_string)

    return vector_string


def normalised_compression_distance(string_1, string_2, compressed_2_size):
    compressed_1_size = None
    compressed_1_2_size = None
    encoded_string_1 = string_1.encode()
    encoded_string_1_2 = (string_1+string_2).encode()

    if zip_type == "zlib":
        compressed_1_size = len(zlib.compress(encoded_string_1))
        compressed_1_2_size = len(zlib.compress(encoded_string_1_2))
    elif zip_type == "gzip":
        compressed_1_size = len(gzip.compress(encoded_string_1))
        compressed_1_2_size = len(gzip.compress(encoded_string_1_2))
    elif zip_type == "bz2":
        compressed_1_size = len(bz2.compress(encoded_string_1))
        compressed_1_2_size = len(bz2.compress(encoded_string_1_2))
    else:
        ValueError(f"{zip_type} not a valid format")

    ncd = (compressed_1_2_size - min(compressed_1_size, compressed_2_size)) / max(compressed_1_size, compressed_2_size)
    return ncd


def get_label(rows: pd.DataFrame, column_name: str, weighted_column_name=None):
    if len(rows[column_name].unique()) == len(rows):
        return rows.iloc[0, 0]
    if weighted_column_name is not None:
        highest_value = rows[weighted_column_name].max()
        weighted_values = highest_value - rows[weighted_column_name]
        weighted_dataframe = pd.DataFrame({column_name: rows[column_name],
                                           "weighted_values": weighted_values})
        sums = weighted_dataframe.groupby(by=column_name).sum().reset_index()
        label = sums[sums["weighted_values"] == sums["weighted_values"].max()][column_name].iloc[0]
    else:
        most_frequents = rows[column_name].mode()
        label = most_frequents[0]

    return label


def get_accuracy_and_confusion_matrix(test_labels, predicted_labels):
    print(f"Accuracy = {accuracy_score(test_labels, predicted_labels)}")
    # ConfusionMatrixDisplay.from_predictions(test_labels, predicted_labels,
    #                                         labels=test_labels.unique(), xticks_rotation="vertical")
    # plt.show()
    # plt.close()


def classify_pokemon(gen="all", weighting=None, k=1):
    start = timeit.default_timer()
    train_data = pd.DataFrame(None)
    test_data = pd.DataFrame(None)
    if gen == "all":
        data: pd.DataFrame = read_data_from_csv(csv_full_data_file)
        data.drop(columns="Type2", inplace=True)
        # print(data["Type1"].value_counts())
        data.drop(data[(data["Type1"] != "Normal") & (data["Type1"] != "Bug") & (data["Type1"] != "Water") &
                       (data["Type1"] != "Grass")].index, inplace=True)
        train_data, test_data, val_data = split_dataframe_to_train_test(data, "Type1", 0.3, random=0)
        train_data.reset_index(drop=True, inplace=True)
        test_data.reset_index(drop=True, inplace=True)
    elif gen == "few":
        train_data: pd.DataFrame = read_data_from_csv(csv_data_1_file)
        test_data: pd.DataFrame = read_data_from_csv(csv_data_2_file)
        # Remove data instances where the Type 1 values are not Normal, Poison, Water, or Grass.
        train_data.drop(train_data[(train_data["Type1"] != "Normal") & (train_data["Type1"] != "Poison") &
                                   (train_data["Type1"] != "Water")  & (train_data["Type1"] != "Grass")].index,
                        inplace=True)
        test_data.drop(test_data[(test_data["Type1"] != "Normal") & (test_data["Type1"] != "Poison") &
                                 (test_data["Type1"] != "Water") & (test_data["Type1"] != "Grass")].index,
                       inplace=True)
        train_data.reset_index(drop=True, inplace=True)
        test_data.reset_index(drop=True, inplace=True)
    else:
        ValueError(f"{gen} not a valid parameter")
        return
    ncds = pd.DataFrame(data=train_data["Type1"])
    # Current dataframe has columns: [Name, Type1, ImgStr]
    train_data["ImgStr"] = train_data["Name"].apply(img_vectorise)
    test_data["ImgStr"] = test_data["Name"].apply(img_vectorise)

    predicted_labels = []

    # For each instance of the test data
    for i in range(len(test_data)):
        # Get the string representation of the test data image
        string: str = test_data.iloc[i, 2]
        # Encode the string
        encoded_string = string.encode()

        # Compress the string to get the length of the compressed string
        if zip_type == "zlib":
            compressed_encoded_string = len(zlib.compress(encoded_string))
        elif zip_type == "gzip":
            compressed_encoded_string = len(gzip.compress(encoded_string))
        else:
            compressed_encoded_string = len(bz2.compress(encoded_string))

        # Calculate the NCDs of the instance and all train data
        ncds["NCD"] = train_data["ImgStr"].apply(normalised_compression_distance, string_2=string,
                                                 compressed_2_size=compressed_encoded_string)
        smallest_rows = ncds.nsmallest(k, "NCD", keep="all")
        # print(f"\n{data_2.iloc[i, 0]} - {data_2.iloc[i, 1]}")
        # print(smallest_rows)
        predicted_labels.append(get_label(smallest_rows, column_name="Type1", weighted_column_name=weighting))

    stop = timeit.default_timer()
    print('Time taken: ', stop - start)
    get_accuracy_and_confusion_matrix(test_data["Type1"], predicted_labels)
    return 0


def main():
    for k in range(1, 13, 2):
        # print(f"Gen 1 and 2 with majority voting classification using {k} nearest neighbours")
        # classify_pokemon(gen="few", weighting=None, k=k)
        # print(f"Gen 1 and 2 with weighted voting classification using {k} nearest neighbours")
        # classify_pokemon(gen="few", weighting="NCD", k=k)
        # print(f"All gens with majority voting classification using {k} nearest neighbours")
        # classify_pokemon(gen="all", weighting=None, k=k)
        print(f"All gens with weighted voting classification using {k} nearest neighbours")
        classify_pokemon(gen="all", weighting="NCD", k=k)


if __name__ == '__main__':
    main()

