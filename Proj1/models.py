import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineNetwork(nn.Module):
    def __init__(self):
        super(BaselineNetwork, self).__init__()
        self.conv1 = nn.Conv2d(2, 16 * 2, 2, 1)  # 2 input channels (one for each digit image), each is 14 x 14
        self.conv2 = nn.Conv2d(16 * 2, 32 * 2, 2, 1)  # 32 input channels, each is 13 x 13
        self.fc1 = nn.Linear(32 * 2 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class Net2(nn.Module):
    def __init__(self, aux_loss = False):
        
        super(Net2, self).__init__()
        self.aux_loss = aux_loss
        
        self.conv1 = nn.Conv2d(2, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.fc1 = nn.Linear(2 * 1024, 128)
        self.fc2 = nn.Linear(128, 2)
        
        if self.aux_loss:
            self.fc1_aux = nn.Linear(1600, 128)
            self.fc2_aux = nn.Linear(128, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        
        if self.aux_loss:
            x_aux = F.max_pool2d(x, 2)
            x_aux = torch.flatten(x_aux, 1)
            x_aux = self.fc1_aux(x_aux)
            x_aux = F.relu(x_aux)
            x_aux = self.fc2_aux(x_aux)
            output_aux = F.softmax(x_aux, dim = 1)
            
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.softmax(x, dim=1)
        
        if self.aux_loss:
            return output, output_aux
        else:
            return output


class Net3(nn.Module):
    def __init__(self, aux_loss = False):
        
        super(Net3, self).__init__()
        self.aux_loss = aux_loss
        
        self.conv1 = nn.Conv2d(2, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        self.conv4 = nn.Conv2d(64, 128, 3, 1)
        self.fc1 = nn.Linear(2 * 576, 128)
        self.fc2 = nn.Linear(128, 2)
        
        if self.aux_loss:
            self.fc1_aux = nn.Linear(800, 128)
            self.fc2_aux = nn.Linear(128, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        
        if self.aux_loss:
            x_aux = F.max_pool2d(x, 2)
            x_aux = torch.flatten(x_aux, 1)
            x_aux = self.fc1_aux(x_aux)
            x_aux = F.relu(x_aux)
            x_aux = self.fc2_aux(x_aux)
            output_aux = F.softmax(x_aux, dim = 1)
            
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.softmax(x, dim=1)
        
        if self.aux_loss:
            return output, output_aux
        else:
            return output


class BaselineNetwork2(nn.Module):
    def __init__(self):
        super(BaselineNetwork2, self).__init__()
        self.fc1 = nn.Linear(2 * 14 * 14, 2)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 16 * 1, 2, 1)  # 2 input channels (one for each digit image), each is 14 x 14
        self.conv2 = nn.Conv2d(16 * 1, 32 * 1, 2, 1)  # 32 input channels, each is 13 x 13
        self.fc1 = nn.Linear(32 * 1 * 12 * 12, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10 * 2, 2, bias=False)

    def forward(self, x):
        x1 = x[:, 0:1, :, :]
        x2 = x[:, 1:2, :, :]

        x1 = self.conv1(x1)
        x1 = F.relu(x1)
        x1 = self.conv2(x1)
        x1 = torch.flatten(x1, 1)
        x1 = self.fc1(x1)
        x1 = F.relu(x1)
        x1 = self.fc2(x1)
        x1 = F.relu(x1)

        x2 = self.conv1(x2)
        x2 = F.relu(x2)
        x2 = self.conv2(x2)
        x2 = torch.flatten(x2, 1)
        x2 = self.fc1(x2)
        x2 = F.relu(x2)
        x2 = self.fc2(x2)
        x2 = F.relu(x2)

        x = torch.cat((x1, x2), 1)

        output = self.fc3(x)
        return output
    
    
class DigitNetwork(nn.Module):  # This network tries to predict a singel digit
    def __init__(self):
        super(DigitNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 16 * 1, 2, 1)  # 2 input channels (one for each digit image), each is 14 x 14
        self.conv2 = nn.Conv2d(16 * 1, 32 * 1, 2, 1)  # 32 input channels, each is 13 x 13
        self.fc1 = nn.Linear(32 * 1 * 12 * 12, 64)
        self.fc2 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    
class SiameseNetwork2(nn.Module):
    def __init__(self):
        super(SiameseNetwork2, self).__init__()
        self.digit = DigitNetwork()
        self.fc3 = nn.Linear(10 * 2, 2, bias=False)
        
    def digit_pred(self, x):
        x = self.digit(x)
        return x
        
    def forward(self, x):
        x1 = x[:, 0:1, :, :]
        x2 = x[:, 1:2, :, :]

        x1 = self.digit_pred(x1)
        x1 = F.relu(x1)
        x2 = self.digit_pred(x2)
        x2 = F.relu(x2)
        
        x = torch.cat((x1, x2), 1)

        output = self.fc3(x)
        return output

