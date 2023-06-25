import pandas as pd
import torch
import pickle
import numpy as np
from dataloading import CustomVisionDataset
from torch.utils.data import DataLoader
from collections import Counter
from utils.utils import softmax
from sklearn.metrics import f1_score

def test(test_path, modelpath, aggregating_method='majority_vote'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Restore model
    with open(modelpath, "rb") as input_file:
        model = pickle.load(input_file)
    model = model.to(device)
    test = pd.read_csv(test_path)
    mean = [0.20610136438155038]
    std = [0.17367093735484107]
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
    #print(grouped_preds)
    #print(grouped_y_true)
    #print(grouped_posteriors)
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
            print(posteriors)
            for image_posteriors in posteriors:
                # Apply softmax to the array
                softmax_array = softmax(image_posteriors)
                new_posteriors.append(softmax_array)
            print(np.array(new_posteriors))
            # Compute the element-wise mean along axis 0 (column-wise)
            mean_axis_0 = np.mean(np.array(new_posteriors), axis=0)
            print(mean_axis_0)
            print(np.argmax(mean_axis_0))
            final_preds.append(np.argmax(mean_axis_0))
    #elif aggregating_method=='average_probability':



    final_test_f1 = f1_score(final_true, final_preds, average='macro')
    print(final_test_f1)



test("utils/test.csv", "model.pt", "majority_vote")