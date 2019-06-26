#  The first file, train.py, will train a new network on a dataset and save the model as a checkpoint. 
import data_processing,load_model,argparse
from workspace_utils import active_session

def train_model(train_dataloader,test_dataloader,epochs):
    pass

# Prints out training loss, validation loss, and validation accuracy as the network trains

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory', help="The location of the files used to train the model.")
    parser.add_argument('-a','--arch', help="Name of pretrained ImageNet model.",dest="arch",default="vgg16")
    parser.add_argument('-hid','--hidden_units', help="No. Hidden Units",dest="hidden_units",default=4096, type=int)
    parser.add_argument('-o','--output_units', help="No. Output Units",dest="output_units",default=102, type=int)
    


    args = parser.parse_args()

    # Loading dataloaders from data directory
    training_files = args.data_directory
    train_dataloader,valid_dataloader,test_dataloader = data_processing.load_images(training_files)
    # Loading Model
    model = args.arch
    hidden_units = args.hidden_units
    output_units = args.output_units
    pretrained_model = load_model.select_pretrained_model(model,hidden_units,output_units)


    
if __name__ == '__main__':
    main()

