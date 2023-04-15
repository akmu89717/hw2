import torch
import torch.nn as nn
import torchvision.models as models

class MyNet(nn.Module): 
    def __init__(self):
        super(MyNet, self).__init__()
               
        self.SEQ1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 3, padding = 1), 
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.1))

        self.SEQ2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size = 3, padding = 1), 
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),		
            nn.Dropout(0.1))

        self.SEQ3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),	 
            nn.ReLU(),
            nn.BatchNorm2d(128))

        self.SEQ4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size = 3, padding=1), 
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),		
            nn.Dropout(0.5))

        self.SEQ5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = 3, padding=1), 
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(0.5))

        self.SEQ6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size = 3, padding=1), 
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.AvgPool2d(8, 8)) 	
        
        self.fc = nn.Linear(256, 10)    

        pass

    def forward(self, x):

        x = self.SEQ1(x)
        x = self.SEQ2(x)
        x = self.SEQ3(x)
        x = self.SEQ4(x)
        x = self.SEQ5(x)
        x = self.SEQ6(x)

        x = x.view(-1,  256)  
        x = self.fc(x)

        pass
        return x
    
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        weights = models.ResNet18_Weights.IMAGENET1K_V1
        self.resnet = models.resnet18(weights)    
        self.resnet.conv1=nn. Conv2d(3, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.resnet.maxpool=Identity()
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 10)

    def forward(self, x):
        return self.resnet(x)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
if __name__ == '__main__':
    model = ResNet18()
    print(model)
