import matplotlib.pyplot as plt
import torch
from CNN import Net
from dataloading import CustomVisionDataset
from torch.utils.data import DataLoader
import pandas as pd
from utils.utils import caclulate_mean_std
import numpy as np
from sklearn.metrics import f1_score
import random
from utils.utils import EarlyStopper, progress
from copy import deepcopy
import math
import time
import pickle

train_path = "utils/train.csv"
val_path = "utils/valid.csv"



def train(train_path, val_path):
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True

    train = pd.read_csv(train_path)
    valid = pd.read_csv(val_path)

    mean, std = caclulate_mean_std(train)


    train_set = CustomVisionDataset(train, mean, std)
    eval_set = CustomVisionDataset(valid, mean, std)


    train_loader = DataLoader(train_set, batch_size=100, shuffle=True, drop_last=True, num_workers=4)
    val_loader = DataLoader(eval_set, batch_size=100, shuffle=True, drop_last=True, num_workers=4)

    net = Net(224, 224, 3, 4, 2, 0.5)

    # use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE: {device}")

    net.to(device)


    ##################################
    # TRAINING PIPELINE
    ##################################
    optimizer = torch.optim.AdamW(params=net.parameters(),
                                  lr=0.002,
                                  weight_decay=.02)
    loss_function = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           'min',
                                                           verbose=True)
    early_stopper = EarlyStopper(patience=10, min_delta=1e-5)

    epochs = 50

    all_epochs = []
    val_loss = 0
    all_train_loss, all_metric_training = [], []
    all_metric_training = []
    all_metric_validation = []
    all_valid_comparison_metric = []
    all_valid_loss = []
    best_model = None
    best_model_epoch = 0
    comparison_metric_max = 0
    # Train CNN
    for epoch in range(1, epochs + 1):  # loop over the dataset multiple times
        scheduler.step(epoch=epoch)
        all_epochs.append(epoch)
        ############## Training ##############
        running_loss = 0.0

        correct_train = 0


        net.train()
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data


            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            labels_cpu = labels.detach().clone().to('cpu').numpy()
            # Get accuracy
            correct_train += sum([int(a == b)
                            for a, b in zip(labels_cpu,
                                            np.argmax(outputs.detach().clone().to('cpu').numpy(),
                                                      axis=1))])
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # print statistics
            progress(loss=running_loss,
                     epoch=epoch,
                     batch=i,
                     batch_size=train_loader.batch_size,
                     dataset_size=len(train_loader.dataset))
        # print statistics
        progress(loss=running_loss / len(train_loader),
                 epoch=epoch,
                 batch=i,
                 batch_size=train_loader.batch_size,
                 dataset_size=len(train_loader.dataset))
        score = correct_train / len(train_loader.dataset) * 100

        # Store statistics for later usage
        all_train_loss.append(running_loss/len(train_loader))
        all_metric_training.append(score)

        ############## Validation ##############

        net.eval()

        correct = 0
        loss_aggregated = 0
        with torch.no_grad():
            pred_all = []
            actual_labels = []
            for j, val_data in enumerate(val_loader):
                val_X, val_y = val_data

                val_X = val_X.to(device)
                val_y = val_y.to(device)

                outputs = net(val_X)

                v_loss = loss_function(outputs, val_y)
                val_loss += v_loss.item()

                preds = outputs.data.max(1, keepdim=True)[1]
                y_pred = np.argmax(outputs.detach().clone().to('cpu').numpy(), axis=1)
                pred_all.append(y_pred)

                labels_cpu = val_y.detach().clone().to('cpu').numpy()
                actual_labels.append(labels_cpu)
                correct += preds.eq(val_y.view_as(preds)).cpu().sum().item()
                loss_aggregated += v_loss.item() * val_X.size(0)
            val_loss = loss_aggregated / (len(val_loader) * val_loader.batch_size)
            score = correct / len(val_loader.dataset)
            labels = [item for sublist in actual_labels for item in sublist]
            preds = [item for sublist in pred_all for item in sublist]
            comparison_metric = f1_score(labels, preds, average='macro')
            if epoch % 5 == 0:
                # Print some stats
                print('\nValidation results for epoch {}:'.format(epoch))

                print('    --> loss: {}'.format(
                    round(val_loss, 4)))
                print('    --> accuracy: {}'.format(round(score, 4)))
                print('    --> f1 score: {}'.format(round(comparison_metric, 4)))
            # Store statistics for later usage
            all_valid_loss.append(val_loss)
            all_metric_validation.append(score)
            all_valid_comparison_metric.append(comparison_metric)
        if (best_model is None) or (comparison_metric > comparison_metric_max + 1e-5):
            comparison_metric_max = comparison_metric
            best_model = deepcopy(net).to('cpu')
            best_model_epoch = epoch
        if epoch != 1:
            if early_stopper.early_stop(comparison_metric):
                print(f'\nResetting model to epoch {best_model_epoch}.')
                break

        net.to('cpu')
        best_model = best_model.to(device)
    print('Finished Training')
    print('All validation accuracies: {} \n'.format(all_metric_validation))
    best_index = all_valid_comparison_metric.index(max(all_valid_comparison_metric))
    print('Best index, best model epoch: ', best_index, best_model_epoch)
    best_model_acc = all_metric_validation[best_index]
    print('Best model\'s validation accuracy: {}'.format(best_model_acc))
    best_model_f1 = all_valid_comparison_metric[best_index]
    print('Best model\'s validation f1 score: {}'.format(best_model_f1))
    best_model_loss = all_valid_loss[best_index]
    print('Best model\'s validation loss: {}'.format(best_model_loss))
    timestamp = time.ctime()
    timestamp = timestamp.replace(" ", "_")
    ofile = f"{best_model.__class__.__name__}_{best_model_epoch}_{timestamp}.pt"
    print(f"\nSaving model to: {ofile}\n")
    best_model = best_model.to("cpu")
    with open(ofile, "wb") as output_file:
        pickle.dump(best_model, output_file)
    plt.figure(figsize=(16, 6))
    plt.plot(all_epochs, all_train_loss, '-o', label='Training loss')
    plt.plot(all_epochs, all_valid_loss, '-o', label='Validation loss')
    plt.legend()
    plt.title('Learning curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(all_epochs, rotation=90)
    plt.tight_layout()

    plt.savefig("losses.png")

    plt.show()

if __name__ == '__main__':
    train()