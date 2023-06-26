import torch
from torchvision import models
from training import train_and_validate, optuna_tune
import argparse


def fine_tune(train_path, validation_path):
    # Load the pretrained DenseNet model
    model = models.densenet121(pretrained=True)
    num_features = model.classifier.in_features

    # Replace the last fully connected layer for binary classification
    model.classifier = torch.nn.Linear(num_features, 2)
    ### uncomment the below comment if you want to tune the entire model instead of just the last linear layer
    '''# Set requires_grad=True for all parameters
    for param in model.parameters():
        param.requires_grad = True'''

    '''
    # Unfreeze the last n layers
    n = 5  # Number of layers to unfreeze

    for param in model.features[-n:].parameters():
        param.requires_grad = True
    '''
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_and_validate(train_path, validation_path, mean, std, model, use_optuna=False, rgb=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', required=True,
                        type=str, help='Input train csv')
    parser.add_argument('-v', '--validation', required=True,
                        type=str, help='Input valid csv')

    args = parser.parse_args()

    # Get argument
    train_path = args.train
    validation_path = args.validation

    fine_tune(train_path, validation_path)
