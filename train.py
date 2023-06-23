import matplotlib.pyplot as plt
import torch
from new_CNN import Net
from dataloading import CustomVisionDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import pandas as pd
from utils.load_dataset import load_split_data
from utils.utils import caclulate_mean_std




data_path = "new_dataset.csv"
def train(data_path):
    train, valid = load_split_data(data_path)

    mean, std = caclulate_mean_std(train)


    train_set = CustomVisionDataset(train, mean, std)
    eval_set = CustomVisionDataset(valid, mean, std)


    train_loader = DataLoader(train_set, batch_size=100, shuffle=True, drop_last=True, num_workers=4)
    val_loader = DataLoader(eval_set, batch_size=100, shuffle=True, drop_last=True, num_workers=4)

    net = Net()

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

    epochs = 30
    batches = 0
    train_losses = list()
    val_losses = list()
    val_acces = list()
    batch_lst = list()

    batches = 0
    val_loss = 0

    # Train CNN
    for epoch in range(1, epochs + 1):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 2000 mini-batches

                net.eval()

                correct = 0

                for j, val_data in enumerate(val_loader):
                    val_X, val_y = val_data

                    val_X = val_X.cuda()
                    val_y = val_y.cuda()

                    outputs = net(val_X)

                    v_loss = loss_function(outputs, val_y)
                    val_loss += v_loss.item()

                    preds = outputs.data.max(1, keepdim=True)[1]

                    correct += preds.eq(val_y.view_as(preds)).cpu().sum().item()

                log = f"epoch: {epoch} {i + 1} " \
                      f"train_loss: {running_loss / 100:.3f} " \
                      f"val_loss: {val_loss / 100:.3f} " \
                      f"Val Acc: {correct / len(val_loader.dataset):.3f}"

                train_losses.append(running_loss / 100)
                val_losses.append(val_loss / 100)
                val_acces.append(correct / len(val_loader.dataset))
                batches += 100
                batch_lst.append(batches)

                val_loss = 0

                print(log)

                running_loss = 0.0

                net.train()

        print('Finished Training')
        plt.figure(figsize=(16, 6))
        plt.plot(batch_lst, train_losses, '-o', label='Training loss')
        plt.plot(batch_lst, val_losses, '-o', label='Validation loss')
        plt.legend()
        plt.title('Learning curves')
        plt.xlabel('Batches')
        plt.ylabel('Loss')
        plt.xticks(batch_lst, rotation=90)
        plt.tight_layout()

        plt.savefig("result.png")

        plt.show()
