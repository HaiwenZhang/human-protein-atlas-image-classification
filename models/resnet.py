import torch
import logging
from torchvision import models


class Net(torch.nn.Module):

    def __init__(self, params):
        super(Net, self).__init__()
        self.pretrained_model = models.resnet50(pretrained=params.use_pretrained)

        self.base_parameters = self.pretrained_model.layer4.parameters()
        self.set_parameter_requires_grad(self.pretrained_model,
                                    params.feature_extracting,
                                    self.base_parameters)
        num_ftrs = self.pretrained_model.fc.in_features
        self.pretrained_model.fc = torch.nn.Linear(num_ftrs, params.num_labels)
        self.pretrained_model.fc.requires_grad = True
        self.input_size = 224
        
        

    def forward(self, x):
        
        x = self.pretraied_model(x)


    def image_size(self):
        return self.input_size

    def base_parameters(self):
        return self.base_parameters

    def last_parameters(self):
        return self.pretrained_model.fc.parameters()

    def set_parameter_requires_grad(model, feature_extracting, base_parameters):
        if feature_extracting:
            logging.info("freeze model parameters")
            ignored_params = list(map(id, base_parameters))
            for p in model.parameters():
                if id(p) not in ignored_params:
                    p.requires_grad = False
                else:
                    p.requires_grad = True

