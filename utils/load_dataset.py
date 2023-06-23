import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import matplotlib.pylab as plt


train_images_paths = "/Users/sofiaeleftheriou/datasets/MURA-v1.1/train_image_paths.csv"
train_labels_path = "/Users/sofiaeleftheriou/datasets/MURA-v1.1/train_labeled_studies.csv"

def load_data():

    data = pd.read_csv(train_images_paths, header=None)
    labels = pd.read_csv(train_labels_path, header=None)
    final_x = []
    final_y = []
    ids = []


    labels_dict = labels.set_index(0)[1].to_dict()
    for index, row in data.iterrows():
        image_path = str(row[0])
        id = image_path.split("/")[-3]
        ids.append(id)
        study_path = "/".join(image_path.split("/")[:-1]) + "/"
        final_x.append(image_path)
        final_y.append(labels_dict[study_path])
    data = {'path': final_x,
            'label': final_y,
            'id': ids}

    # Create DataFrame
    df = pd.DataFrame(data)
    df.to_csv("new_dataset.csv", index=False)


def check_modality_distribution(train, test):
    modalities_distribution_train = {"XR_ELBOW": 0, "XR_FINGER": 0, "XR_FOREARM": 0, "XR_HAND": 0,
    "XR_HUMERUS": 0, "XR_SHOULDER": 0, "XR_WRIST": 0}
    modalities_distribution_test = {"XR_ELBOW": 0, "XR_FINGER": 0, "XR_FOREARM": 0, "XR_HAND": 0,
                                    "XR_HUMERUS": 0, "XR_SHOULDER": 0, "XR_WRIST": 0}
    for index, row in train.iterrows():
        image_path = str(row['path'])
        modality = image_path.split("/")[-4]
        modalities_distribution_train[modality] += 1

    plt.bar(range(len(modalities_distribution_train)), list(modalities_distribution_train.values()), align='center')
    plt.xticks(range(len(modalities_distribution_train)), list(modalities_distribution_train.keys()))
    # # for python 2.x:
    # plt.bar(range(len(D)), D.values(), align='center')  # python 2.x
    # plt.xticks(range(len(D)), D.keys())  # in python 2.x
    plt.savefig('train_modalities_distribution.png')
    plt.clf()
    for index, row in test.iterrows():
        image_path = str(row['path'])
        modality = image_path.split("/")[-4]
        modalities_distribution_test[modality] += 1

    plt.bar(range(len(modalities_distribution_test)), list(modalities_distribution_test.values()), align='center')
    plt.xticks(range(len(modalities_distribution_test)), list(modalities_distribution_test.keys()))
    # # for python 2.x:
    # plt.bar(range(len(D)), D.values(), align='center')  # python 2.x
    # plt.xticks(range(len(D)), D.keys())  # in python 2.x
    plt.savefig('valid_modalities_distribution.png')


def check_labels_distribution(train, test):
    labels_distribution_train = {"0": 0, "1": 0}
    labels_distribution_test = {"0": 0, "1": 0}
    plt.clf()
    for index, row in train.iterrows():
        label = str(row['label'])
        labels_distribution_train[label] += 1

    plt.bar(range(len(labels_distribution_train)), list(labels_distribution_train.values()), align='center')
    plt.xticks(range(len(labels_distribution_train)), list(labels_distribution_train.keys()))
    # # for python 2.x:
    # plt.bar(range(len(D)), D.values(), align='center')  # python 2.x
    # plt.xticks(range(len(D)), D.keys())  # in python 2.x
    plt.savefig('train_labels_distribution.png')
    plt.clf()
    for index, row in test.iterrows():
        label = str(row['label'])
        labels_distribution_test[label] += 1

    plt.bar(range(len(labels_distribution_test)), list(labels_distribution_test.values()), align='center')
    plt.xticks(range(len(labels_distribution_test)), list(labels_distribution_test.keys()))
    # # for python 2.x:
    # plt.bar(range(len(D)), D.values(), align='center')  # python 2.x
    # plt.xticks(range(len(D)), D.keys())  # in python 2.x
    plt.savefig('valid_labels_distribution.png')

def initial_labels_distribution(df):
    labels_distribution_initial = {"0": 0, "1": 0}
    plt.clf()
    for index, row in df.iterrows():
        label = str(row['label'])
        labels_distribution_initial[label] += 1

    plt.bar(range(len(labels_distribution_initial)), list(labels_distribution_initial.values()), align='center')
    plt.xticks(range(len(labels_distribution_initial)), list(labels_distribution_initial.keys()))
    plt.savefig('initial_labels_distribution.png')

def load_split_data(data_path):
    df = pd.read_csv(data_path)
    splitter = GroupShuffleSplit(test_size=.20, n_splits=2, random_state=7)
    split = splitter.split(df, groups=df['id'])
    train_inds, test_inds = next(split)

    train = df.iloc[train_inds]
    test = df.iloc[test_inds]
    print(train)
    print(test)
    check_modality_distribution(train, test)
    check_labels_distribution(train, test)
    initial_labels_distribution(df)
    return train, test


