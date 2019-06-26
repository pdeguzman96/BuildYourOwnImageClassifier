from torchvision import models
from torch import nn
from collections import OrderedDict

def select_pretrained_model(model_name,hidden_units,no_output_categories):
    '''
    Inputs: 
    model_name: name of pretrained model trained on ImageNet
        See documentation for models: https://pytorch.org/docs/master/torchvision/models.html
    hidden_units: no. units in hidden layer
    no_output_categories: no. of output units

    Returns: Pretrained model with updated classifier to be trained on new data

    Feature parameters for the pretrained model are frozen. 
    
    Classifier to be trained is replaced to work with 220x220 PIL images.
    '''
    pretrained_model = getattr(models,model_name)(pretrained=True)
    for param in pretrained_model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(18432,hidden_units)),
                                ('relu', nn.ReLU()),
                                ('fc2', nn.Linear(hidden_units,no_output_categories)),
                                ('output', nn.LogSoftmax(dim=1))
                                ]))
    pretrained_model.classifier = classifier

    return pretrained_model

if __name__=='__main__':
    # Test load
    model = select_pretrained_model('vgg16',4096,102)
    if model:
        print("Model successfully Loaded")
