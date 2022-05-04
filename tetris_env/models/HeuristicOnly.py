import torch
import torch.nn as nn

class HeuristicNetwork(nn.Module):

    def __init__(self):

        super(HeuristicNetwork, self).__init__()
        
        self.fc = nn.Sequential(
                nn.Linear(11, 64),
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


        return self.fc(input['heuristic'])
