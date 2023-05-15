import torch.nn as nn

class NaturalSceneClassification(ImageClassificationBase):
    def __init__(self, height, width, config, num_cnn_layers, num_linear_layers, dropout):
        super().__init__() 

        layers = {}        

        input_channels = 1
        for layer in range(1, num_cnn_layers+1):
            layers[layer] = nn.Sequential(
                nn.Conv2d(input_channels, config['conv_' + layer]['num_channels'], config['conv_' + layer]['kernel_size'],
                          stride=config['conv_' + layer]['stride'], padding=config['conv_' + layer]['padding']),
                nn.BatchNorm2d(config['conv_' + layer]['num_channels']),
                nn.ReLU(),
                nn.MaxPool2d(config['max_pool_' + layer]['kernel_size'], stride = config['max_pool_' + layer]['stride'])
            )
            input_channels =  config['conv_' + layer]['num_channels'] # for next iteration

            (height, width) = self.conv_size(height, width, config['conv_' + layer]['kernel_size'], config['conv_' + layer]['stride']) 
            (height, width) = self.max_pool_size(height, width, config['max_pool_' + layer]['kernel_size'], config['max_pool_' + layer]['padding'])



        linear_input_size = height * width * config['conv_' + num_cnn_layers]['num_channels']
        for layer in range(num_cnn_layers+1, num_cnn_layers+num_linear_layers+1):
            linear_counter = layer - num_cnn_layers
            layers[layer] = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(linear_input_size , config['linear_' + linear_counter]['size']),
                nn.ReLU()
            )
            linear_input_size = config['linear_' + linear_counter]['size']

        layers[layer] = nn.Sequential(
            nn.Linear(linear_input_size, config['linear_' + (linear_counter + 1)]['size'])
        )

        

            linear_input_size = 
            





        self.linear_layer1 = nn.Sequential(
                nn.Dropout(0.75),
                nn.Linear(self.calc_out_size(), 1024),
                nn.LeakyReLU()
            )
        self.network = nn.Sequential(
            
            nn.Conv2d(1, 32, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(32,64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(128 ,128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(256,256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Flatten(),
            nn.Linear(82944,1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512,6)
        )
    
    def forward(self, xb):
        return self.network(xb)

        
    def max_pool_size(self, height, width, kernel_size, stride):
        return ((height - kernel_size) / stride + 1, (width - kernel_size) / stride + 1) 
    
    

    def conv_size(self, height, width, kernel_size, stride, padding):
        return ((height + 2 * padding - kernel_size) / stride + 1, (width + 2 * padding - kernel_size) / stride + 1) 