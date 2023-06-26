import argparse
from training import optuna_tune, train_and_validate
from CNN import Net
def train_CNN(train_path, validation_path, use_optuna):
    mean = [0.20610136438155038]
    std = [0.17367093735484107]
    if use_optuna:
        optuna_tune(train_path, validation_path, mean, std)
    else:
        params = {
            'conv_layers': 2,
            'num_channels': 3,
            'dense_nodes': 2,
            'dropout': 0.22023959319449948
        }
        net = Net(224, 224, params)
        train_and_validate(train_path, validation_path, mean, std, net=net, use_optuna=False)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', required=True,
                        type=str, help='Input train csv')
    parser.add_argument('-v', '--validation', required=True,
                        type=str, help='Input valid csv')
    parser.add_argument('--optuna', required=False, action='store_true',
                        help='use optuna tuner')
    args = parser.parse_args()

    # Get argument
    train_path = args.train
    validation_path = args.validation
    use_optuna = args.optuna
    train_CNN(train_path, validation_path, use_optuna)
