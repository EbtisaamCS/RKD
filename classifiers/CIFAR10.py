import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Classifier(nn.Module):
    def __init__(self,num_classes=10):
        super(Classifier, self).__init__()
        # Load a pre-trained ResNet model
        self.model = models.resnet18(pretrained=True)

        # Correctly configure the first layer to accept 3-channel input
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Adjust the average pooling layer and fully connected layer
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)
 


