import pandas as pd

def create_csv(paths_csv, labels_csv, name):

    data = pd.read_csv(paths_csv, header=None)
    labels = pd.read_csv(labels_csv, header=None)
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
    df.to_csv(name + ".csv", index=False)

    
