import torch
import torch.nn as nn

class HeuristicHoleNetwork(nn.Module):

    def __init__(self):

        super(HeuristicHoleNetwork, self).__init__()
        
        self.fc = nn.Sequential(
                nn.Linear(21, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
        )

    def forward(self, input):
        if len(input['board'].shape) == 3:
            input['heuristic'] = input['heuristic'].unsqueeze(0)
            input['holes'] = input['holes'].unsqueeze(0)

        fc_input = torch.cat((input['heuristic'], input['holes']), dim = 1)

        return self.fc(fc_input)
