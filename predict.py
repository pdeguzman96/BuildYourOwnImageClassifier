import os, glob, json, argparse, logging
from torch import nn
import torch
from collections import OrderedDict
import load_model
import data_processing

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

def print_predictions(probabilities, classes,image,checkpoint,category_names=None,save_results=False):
    '''
    Prints the system output of probabilities.
    '''
    print(image)
    if category_names:
        labels = class_to_label(category_names,classes)
        for i,(ps,ls,cs) in enumerate(zip(probabilities,labels,classes),1):
            print(f'{i}) {ps*100:.2f}% {ls.title()} | Class No. {cs}')
            if save_results:
                logger.info(f'{checkpoint},{image},{i},{ps},{cs},{ls.title()}')      
    else:
        for i,(ps,cs) in enumerate(zip(probabilities,classes),1):
            print(f'{i}) {ps*100:.2f}% Class No. {cs} ')
            if save_results:
                logger.info(f'{checkpoint},{image},{i},{ps},{cs},{category_names}')  
    print('')          

def return_image_files(image_dir):
    '''
    Input: Directory with jpg files to predict
    '''
    cwd = os.getcwd()
    os.chdir(image_dir)
    image_filenames = []
    for file in glob.glob("*.jpg"):
        image_filenames.append(file)
    os.chdir(cwd)
    return image_filenames

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint',help="Path to location of trained model checkpoint.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-i', '--image',help="Path to location of image.")
    group.add_argument('-d', '--dir',help="Path to location of directory.",dest='img_dir')
    parser.add_argument('-t','--top_k',help="No. of top classes to return",type=int,default=5)
    parser.add_argument('-g','--gpu', help="Use GPU (CUDA)?", action="store_true")
    parser.add_argument('-cn','--category_names',help="JSON Category to Label mapping")
    parser.add_argument('-sr','--save_results',help="Save predictions to predictions.csv?", action="store_true")

    args = parser.parse_args()
    image = args.image
    img_dir = args.img_dir
    checkpoint = args.checkpoint
    topk = args.top_k
    category_names = args.category_names
    if args.gpu:
        device = 'cuda'
    else:
        device = 'cpu'    
        save_results = args.save_results
    os.system('clear')
    if save_results:
        global logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s,%(message)s')
        file = 'predictions.csv'
        file_handler = logging.FileHandler(file)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        print(f"Loading {checkpoint}. All predictions will be logged & saved in {file}.\n")
    else:    
        print(f"Loading {checkpoint}...\n")

    model_arch,input_units, output_units, hidden_units, state_dict, class_to_idx = load_checkpoint(checkpoint,device)
    model = load_model.select_pretrained_model(model_arch,hidden_units,output_units)
    model.load_state_dict(state_dict)

    index_mapping = dict(map(reversed, class_to_idx.items()))
    

    if image: 
        print("Prediction...\n")
        probabilities,classes = predict(image,model,index_mapping,topk,device)
        if category_names:
            # print(image.split('/')[-1])
            print_predictions(probabilities,classes,image.split('/')[-1],checkpoint,category_names,save_results=save_results)      
        else:
            # print(image.split('/')[-1])
            print_predictions(probabilities,classes,image.split('/')[-1],checkpoint,save_results=save_results)          
    elif img_dir:
        print("Predictions...\n")
        image_paths = return_image_files(img_dir)
        for img in image_paths:
            probabilities, classes = predict(img_dir+'/'+img,model,index_mapping,topk,device)
            if category_names:
                # print(img)
                print_predictions(probabilities,classes,img,checkpoint,category_names,save_results=save_results)      
            else:
                # print(img)
                print_predictions(probabilities,classes,img,checkpoint,save_results=save_results)

if __name__=='__main__':
    main()

