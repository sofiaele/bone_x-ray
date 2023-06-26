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
import argparse
import optuna
from torchvision import models


#train_path = "utils/train.csv"
#val_path = "utils/valid.csv"




def train_and_validate(train_path, val_path, mean, std, net=None, params=None, trial=None, use_optuna=False, rgb=False):
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True

    train = pd.read_csv(train_path)
    valid = pd.read_csv(val_path)

    #mean, std = caclulate_mean_std(train)
    #print(mean, std)
    #

    train_set = CustomVisionDataset(train, mean, std, rgb=rgb)
    eval_set = CustomVisionDataset(valid, mean, std, rgb=rgb)


    train_loader = DataLoader(train_set, batch_size=100, shuffle=True, drop_last=True, num_workers=4)
    val_loader = DataLoader(eval_set, batch_size=100, shuffle=True, drop_last=True, num_workers=4)


    # use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE: {device}")

    net.to(device)


    ##################################
    # TRAINING PIPELINE
    ##################################
    if use_optuna:
        optimizer = getattr(torch.optim, params['optimizer_name'])(net.parameters(), lr=params['lr'])
    else:
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
        net.to(device)

        all_epochs.append(epoch)
        ############## Training ##############
        running_loss = 0.0

        correct_train = 0


        net.train()
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, _ = data


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
            running_loss += loss.data.item()
            # print statistics
            progress(loss=loss.data.item(),
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
                #print(j)
                val_X, val_y, _ = val_data

                val_X = val_X.to(device)
                val_y = val_y.to(device)

                outputs = net(val_X)

                v_loss = loss_function(outputs, val_y)

                y_pred = np.argmax(outputs.detach().clone().to('cpu').numpy(), axis=1)
                pred_all.append(y_pred)

                labels_cpu = val_y.detach().clone().to('cpu').numpy()
                actual_labels.append(labels_cpu)

                # Get accuracy
                correct += sum([int(a == b)
                                for a, b in zip(labels_cpu, y_pred)])

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
        scheduler.step(val_loss)

        if use_optuna:
            # Add prune mechanism
            trial.report(comparison_metric, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if epoch != 1:
            if early_stopper.early_stop(comparison_metric):
                print(f'\nResetting model to epoch {best_model_epoch}.')
                break

        net.to('cpu')
        best_model = best_model.to(device)
    print('Finished Training')
    print('All validation accuracies: {} \n'.format(all_metric_validation))
    best_index = all_valid_comparison_metric.index(max(all_valid_comparison_metric))

    best_model_acc = all_metric_validation[best_index]
    print('Best model\'s validation accuracy: {}'.format(best_model_acc))
    best_model_f1 = all_valid_comparison_metric[best_index]
    print('Best model\'s validation f1 score: {}'.format(best_model_f1))
    best_model_loss = all_valid_loss[best_index]
    print('Best model\'s validation loss: {}'.format(best_model_loss))

    if not(use_optuna):
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
        plt.xticks(all_epochs)
        plt.tight_layout()

        plt.savefig("losses.png")

        plt.show()
    return best_model_f1

def objective(trial, train_path, val_path, mean, std):
    global tuner
    params = {
        'conv_layers': trial.suggest_int("conv_layers", 1, 4),
        'num_channels': trial.suggest_int("num_channels", 1, 4),
        'dense_nodes': trial.suggest_int("dense_nodes", 1, 5),
        'dropout': trial.suggest_uniform('dropout', 1e-1, 9e-1),
        'optimizer_name': trial.suggest_categorical("optimizer_name", ["Adam", "RMSprop", "SGD"]),
        'lr': trial.suggest_loguniform("lr", 1e-5, 1e-1)
    }

    model = Net(224, 224, params)

    f1 = train_and_validate(train_path, val_path, mean, std, model, params, trial, use_optuna=True)

    return f1


def optuna_tune(train_path, val_path, mean, std):
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner())
    # Wrap the objective inside a lambda and call objective inside it
    func = lambda trial: objective(trial, train_path, val_path, mean, std)
    study.optimize(func, n_trials=30)

    '''timestamp = time.ctime()
    timestamp = timestamp.replace(" ", "_")
    ofile = f"{best_booster.__class__.__name__}_{timestamp}.pt"
    print(f"\nSaving model to: {ofile}\n")
    best_model = best_booster.to("cpu")
    with open(ofile, "wb") as output_file:
        pickle.dump(best_model, output_file)'''

    best_trial = study.best_trial
    print("Best trial:", best_trial)


    for key, value in best_trial.params.items():
        print("{}: {}".format(key, value))
    optuna.visualization.plot_intermediate_values(study)
    optuna.visualization.plot_optimization_history(study)
    optuna.visualization.plot_param_importances(study)

