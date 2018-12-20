import torch.nn as nn
from torchvision import models

def initialize_pretrained_model(model_name, num_labels, feature_extract, use_pretrained=True):
    model_ft = None
    input_size = 0
    if model_name == "resnet50":
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_labels)
        input_size = 224
    elif model_name == "res":
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_labels)
        input_size = 224
    elif model_name == "desenet":
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_labels)
        input_size = 224
    return model_ft, input_size

class MyModel(nn.Module):
    def __init__(self, params):

        super(MyModel, self).__init__()        
        self.pretrained_model, self.input_size =initialize_pretrained_model(params.model_name,
                                                                            params.num_labels,
                                                                            params.feature_extract,
                                                                            params.use_pretrained)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pretrained_model(x)
        # x = self.sigmoid(x)
        return x

    def image_size(self):
        return self.input_size

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
