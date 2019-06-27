import os
import argparse
from torch import nn
from collections import OrderedDict
import load_model
import data_processing
import json
import torch

def load_checkpoint(filepath,device):
    '''
    Inputs...
        Filepath: location of checkpoint
        Device: "gpu" or "cpu"
    
    Returns:
        No. Input units, No. Output units, No. Hidden Units, State_Dict
    '''
    # Loading on GPU If available
    if device=="gpu":
        map_location=lambda device, loc: device.cuda()
    else:
        map_location='cpu'
    checkpoint = torch.load(f=filepath,map_location=map_location)
    return checkpoint['model_arch'],checkpoint['clf_input'], checkpoint['clf_output'], checkpoint['clf_hidden'],checkpoint['state_dict'],checkpoint['model_class_to_idx']

def class_to_label(file,classes):
    '''
    Takes a JSON file containing the mapping from class to label and converts it into a dict.
    '''
    with open(file, 'r') as f:
        class_mapping =  json.load(f)
    labels = []
    for c in classes:
        labels.append(class_mapping[c])
    return labels

def predict(image_path, model,index_mapping, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    - Mapping is the dictionary mapping indices to classes
    '''
    import torch
    pre_processed_image = torch.from_numpy(data_processing.process_image(image_path))
    pre_processed_image = torch.unsqueeze(pre_processed_image,0).to(device).float()
    model.to(device)
    model.eval()
    log_ps = model.forward(pre_processed_image)
    ps = torch.exp(log_ps)
    top_ps,top_idx = ps.topk(topk,dim=1)
    list_ps = top_ps.tolist()[0]
    list_idx = top_idx.tolist()[0]
    classes = []
    model.train()
    for x in list_idx:
        classes.append(index_mapping[x])
    return list_ps, classes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image',help="Path to location of image.")
    parser.add_argument('checkpoint',help="Path to location of trained model checkpoint.")
    parser.add_argument('-t','--top_k',help="No. of top classes to return",dest="top_k",type=int,default=3)
    parser.add_argument('-g','--gpu', help="Use GPU (CUDA)?", action="store_true")
    parser.add_argument('-cn','--category_names',help="JSON Category to Label mapping",dest="category_names")

    args = parser.parse_args()
    image = args.image
    checkpoint = args.checkpoint
    topk = args.top_k
    category_names = args.category_names
    if args.gpu:
        device = 'cuda'
    else:
        device = 'cpu'    

    model_arch,input_units, output_units, hidden_units, state_dict, class_to_idx = load_checkpoint(checkpoint,device)
    model = load_model.select_pretrained_model(model_arch,hidden_units,output_units)
    model.load_state_dict(state_dict)

    index_mapping = dict(map(reversed, class_to_idx.items()))
    probabilities,classes = predict(image,model,index_mapping,topk,device)
    
    if category_names:
        labels = class_to_label(category_names,classes)
    else:
        labels = classes

    os.system('clear')
    print("PREDICTIONS")
    for i,(ps,ls) in enumerate(zip(probabilities,labels),1):
        print(f'{i}) {ps*100:.2f}% {ls.title()} ')

if __name__=='__main__':
    main()