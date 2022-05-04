import torch
import torch.nn as nn

class ConvBasedNetwork(nn.Module):

    def __init__(self):

        super(ConvBasedNetwork, self).__init__()
        
        self.board_conv = nn.Sequential(
                nn.ConstantPad2d(1, 1),
                nn.Conv2d(1, 64, 3, padding='same'),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding='same'),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding='same'),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 1, 3, padding='same'),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten()
        )
        
        self.fc = nn.Sequential(
                nn.Linear(133, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
        )

    def forward(self, input):
        if len(input['board'].shape) == 3:
            input['board'] = input['board'].unsqueeze(0)
            input['other_data'] = input['other_data'].unsqueeze(0)
            input['heuristic'] = input['heuristic'].unsqueeze(0)

        conv_out = self.board_conv(input['board'])
        fc_input = torch.cat((conv_out, input['other_data'], input['heuristic']), dim = 1)


        return self.fc(fc_input)
