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




    
class DigitNetwork_2conv(nn.Module):  # This network tries to predict a single digit
    def __init__(self, dropout=False):
        super(DigitNetwork_2conv, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 2, 1)  # 1 input channel 14 x 14
        self.conv2 = nn.Conv2d(16, 32, 2, 1)  # 16 input channels, each is 13 x 13
        self.fc1 = nn.Linear(32 * 12 * 12, 64)
        self.fc2 = nn.Linear(64, 10)
        self.dropout = dropout
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        if self.dropout: x = F.dropout2d(x, 0.25)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        if self.dropout: x = F.dropout2d(x, 0.25)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    
    
class DigitNetwork_3conv(nn.Module):  # This network tries to predict a single digit
    def __init__(self, dropout=False):
        super(DigitNetwork_3conv, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 2, 1)  # 1 input channel  14 x 14
        self.conv2 = nn.Conv2d(16, 32, 2, 1)  # 16 input channels, each is 13 x 13
        self.conv3 = nn.Conv2d(32, 64, 2, 1)  # 32 input channels, each is 12 x 12
        self.fc1 = nn.Linear(64 * 11 * 11, 64)
        self.fc2 = nn.Linear(64, 10)
        self.dropout = dropout
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        if self.dropout: x = F.dropout2d(x, 0.25)
        x = self.conv2(x)
        if self.dropout: x = F.dropout2d(x, 0.25)
        x = F.relu(x)
        x = self.conv3(x)
        if self.dropout: x = F.dropout2d(x, 0.25)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    

class DigitNetwork_4conv(nn.Module):  # This network tries to predict a single digit
    def __init__(self, dropout=False):
        super(DigitNetwork_4conv, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 2, 1)  # 1 input channel  14 x 14
        self.conv2 = nn.Conv2d(16, 32, 2, 1)  # 16 input channels, each is 13 x 13
        self.conv3 = nn.Conv2d(32, 64, 2, 1)  # 32 input channels, each is 12 x 12
        self.conv4 = nn.Conv2d(64, 128, 2, 1)  # 64 input channels, each is 11 x 11
        self.fc1 = nn.Linear(128 * 10 * 10, 64)
        self.fc2 = nn.Linear(64, 10)
        self.dropout = dropout
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        if self.dropout: x = F.dropout2d(x, 0.25)
        x = self.conv2(x)
        if self.dropout: x = F.dropout2d(x, 0.25)
        x = F.relu(x)
        x = self.conv3(x)
        if self.dropout: x = F.dropout2d(x, 0.25)
        x = F.relu(x)
        x = self.conv4(x)
        if self.dropout: x = F.dropout2d(x, 0.25)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    
class SiameseNetwork(nn.Module):
    def __init__(self, nb_conv, dropout=False, final_bias=False):
        super(SiameseNetwork, self).__init__()
        
        if nb_conv == 2:
            self.digit = DigitNetwork_2conv(dropout=dropout)
        elif nb_conv == 3:
            self.digit = DigitNetwork_3conv(dropout=dropout)
        elif nb_conv == 4:
            self.digit = DigitNetwork_4conv(dropout=dropout)
        else:
            print("ERROR: number of conv layers not supported (use 2, 3 or 4)")

        self.fc = nn.Linear(10 * 2, 2, bias=final_bias)
        
    def digit_pred(self, x):
        x = self.digit(x)
        return x
    
    # Either freeze or unfreeze the learning of the digit prediction part
    def digit_pred_req_grad(self, requires_grad):
        for param in self.digit.parameters():
            param.requires_grad = requires_grad
            
    # Either freeze or unfreeze the learning of the digit comparison part
    def digit_comp_req_grad(self, requires_grad):
        for param in self.fc.parameters():
            param.requires_grad = requires_grad
        
    def forward(self, x):
        x1 = x[:, 0:1, :, :]
        x2 = x[:, 1:2, :, :]

        x1 = self.digit_pred(x1)
        x1 = F.relu(x1)
        x2 = self.digit_pred(x2)
        x2 = F.relu(x2)
        
        x = torch.cat((x1, x2), 1)

        output = self.fc(x)
        return output

