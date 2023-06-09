import pandas as pd
import torch
import pickle
import numpy as np
from dataloading import CustomVisionDataset
from torch.utils.data import DataLoader
from collections import Counter
from utils.utils import softmax
from sklearn.metrics import f1_score
import argparse
from sklearn.metrics import cohen_kappa_score
def test(test_path, modelpath, aggregating_method='majority_vote', extremity_type=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Restore model
    with open(modelpath, "rb") as input_file:
        model = pickle.load(input_file)
    model = model.to(device)
    test = pd.read_csv(test_path)
    print(test)
    if extremity_type!=None:
       test = test.loc[test['path'].str.contains(extremity_type)]
    print(test)
    mean =  [0.4998042153140976]
    std = [0.29930012885944535]
    test_set = CustomVisionDataset(test, mean, std, mode="test", rgb=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True, drop_last=False)

    ############### Test pipeline ###############
    model.eval()

    correct = 0
    # Create empty array for storing predictions and labels
    posteriors = []
    preds = []
    y_true = []
    samples_paths = []

    for index, batch in enumerate(test_loader):
        # Split each batch[index]
        inputs, labels, image_path = batch

        # Transfer to device
        inputs = inputs.to(device)

        labels = labels.to(device)
        outputs = model(inputs)

        # Predict the one with the maximum probability
        predictions = torch.argmax(outputs, -1)
        # Save predictions
        preds.append(predictions.cpu().data.numpy())
        y_true.append(labels.cpu().data.numpy())
        posteriors.append(outputs[0].cpu().detach().numpy())
        study_id = "/".join(image_path[0].split("/")[:-1])
        samples_paths.append(study_id)
    # Get metrics
    #print(np.array(preds).shape)

    posteriors = np.array(posteriors)
    preds = list(np.array(preds).flatten())
    y_true = list(np.array(y_true).flatten())
    samples_paths = list(np.array(samples_paths).flatten())
    values = set(samples_paths)
    grouped_preds = [[preds[index] for index in range(len(preds)) if samples_paths[index] == x] for x in values]
    grouped_y_true = [[y_true[index] for index in range(len(y_true)) if samples_paths[index] == x] for x in values]
    grouped_posteriors = [[posteriors[index] for index in range(len(posteriors)) if samples_paths[index] == x] for x in values]

    if aggregating_method==None:
        final_test_f1 = f1_score(y_true, preds, average='macro')
        final_cohen_score = cohen_kappa_score(preds, y_true)
    else:
        final_true = []
        final_preds = []

        for y_true, preds, posteriors in zip(grouped_y_true, grouped_preds, grouped_posteriors):
            # Count the occurrences of each decision
            counter = Counter(y_true)
            # Find the decision with the highest count
            final_true.append(counter.most_common(1)[0][0])
            if aggregating_method == 'majority_vote':
                counter = Counter(preds)
                # Find the decision with the highest count
                final_preds.append(counter.most_common(1)[0][0])
            elif aggregating_method == 'average_probability':
                new_posteriors = []
                for image_posteriors in posteriors:
                    # Apply softmax to the array
                    softmax_array = softmax(image_posteriors)
                    new_posteriors.append(softmax_array)
                # Compute the element-wise mean along axis 0 (column-wise)
                mean_axis_0 = np.mean(np.array(new_posteriors), axis=0)
                final_preds.append(np.argmax(mean_axis_0))
            elif aggregating_method == 'pos_max':
                new_posteriors = []
                for image_posteriors in posteriors:
                    # Apply softmax to the array
                    softmax_array = softmax(image_posteriors)
                    new_posteriors.append(softmax_array[1])
                max_pos_class = np.max(np.array(new_posteriors))
                if max_pos_class > 0.5:
                    final_preds.append(1)
                else:
                    final_preds.append(0)
        final_test_f1 = f1_score(final_true, final_preds, average='macro')
        final_cohen_score = cohen_kappa_score(final_preds, final_true)


    print("f1: ", final_test_f1)
    print("Cohen score: ", final_cohen_score)



#test("utils/test.csv", "model2.pt", None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test', required=True,
                        type=str, help='Input test csv')
    parser.add_argument('-m', '--model_path', required=True,
                        type=str, help='model path')
    parser.add_argument('--method', required=False, default=None,
                        help='which method to aggregate')
    args = parser.parse_args()

    # Get argument
    test_path = args.test
    model_path = args.model_path
    method = args.method
    test(test_path, model_path, method)