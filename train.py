import os,sys,time,argparse
import data_processing,load_model
from torch import optim,nn
import torch
from workspace_utils import active_session # Provided by Udacity - Purpose: Keep GPU session active 

def train_model(model,train_dataloader,valid_dataloader,test_dataloader,epochs,device,lr,print_every):
    '''
    Trains and validates model on training data and validation data.
    Note: Optimizer used is Adam and Loss function is Negative Log Likelihood Loss
    
    Prints out training loss, training accuracy, validation loss, and validation accuracy as the network trains.
    Additionally, plots training summary via matplotlib.
    '''
    model.to(device)
    optimizer = optim.Adam(model.classifier.parameters(),lr=lr)
    criterion = nn.NLLLoss()
    running_loss = running_accuracy = 0
    validation_losses, training_losses = [],[]

    os.system('clear')
    print("Training model...")
    
    # with active_session():
    for e in range(epochs):
        start_epoch = time.time()
        batches = 0
        # Turning on training mode
        model.train()
        for images,labels in train_dataloader:
            start = time.time()
            batches += 1
            # Moving images & labels to the device
            images,labels = images.to(device),labels.to(device)
            # Pushing batch through network, calculating loss & gradient, and updating weights
            log_ps = model.forward(images)
            loss = criterion(log_ps,labels)
            loss.backward()
            optimizer.step()
            # Calculating metrics
            ps = torch.exp(log_ps)
            top_ps, top_class = ps.topk(1,dim=1)
            matches = (top_class == labels.view(*top_class.shape)).type(torch.FloatTensor)
            accuracy = matches.mean()
            # Resetting optimizer gradient & tracking metrics
            optimizer.zero_grad()
            running_loss += loss.item()
            running_accuracy += accuracy.item()
            # Running the model on the validation set every print_every loops
            if batches%print_every == 0:
                end = time.time()
                training_time = end-start
                start = time.time()
                # Setting metrics
                validation_loss = 0
                validation_accuracy = 0
                # Turning on evaluation mode & turning off calculation of gradients
                model.eval()
                with torch.no_grad():
                    for images,labels in valid_dataloader:
                        images,labels = images.to(device),labels.to(device)
                        log_ps = model.forward(images)
                        loss = criterion(log_ps,labels)
                        ps = torch.exp(log_ps)
                        top_ps, top_class = ps.topk(1,dim=1)
                        matches = (top_class == \
                                labels.view(*top_class.shape)).type(torch.FloatTensor)
                        accuracy = matches.mean()
                        # Tracking validation metrics
                        validation_loss += loss.item()
                        validation_accuracy += accuracy.item()
                
                # Tracking metrics
                end = time.time()
                validation_time = end-start
                validation_losses.append(running_loss/print_every)
                training_losses.append(validation_loss/len(valid_dataloader))
                
                # Printing Results
                print(f'Epoch {e+1}/{epochs} | Batch {batches}')
                print(f'Running Training Loss: {running_loss/print_every:.3f}')
                print(f'Running Training Accuracy: {running_accuracy/print_every*100:.2f}%')
                print(f'Validation Loss: {validation_loss/len(valid_dataloader):.3f}')
                print(f'Validation Accuracy: {validation_accuracy/len(valid_dataloader)*100:.2f}%')
                print(f'Training Time: {training_time:.3f} seconds for {print_every} batches.')
                print(f'Validation Time: {validation_time:.3f} seconds.\n')

                # Resetting metrics & turning on training mode
                running_loss = running_accuracy = 0
                model.train()         
        end_epoch = time.time()
        epoch_time = end_epoch-start_epoch
        print("=======================================")
        print(f'Total Elapsed Time for Epoch {e+1}: {epoch_time:.3f} seconds.')
        print("=======================================\n")

    print("Evaluating Model on Testing Data...\n")
    test_accuracy = 0
    for images,labels in test_dataloader:
        model.eval()
        images,labels = images.to(device),labels.to(device)
        log_ps = model.forward(images)
        ps = torch.exp(log_ps)
        top_ps,top_class = ps.topk(1,dim=1)
        matches = (top_class == labels.view(*top_class.shape)).type(torch.FloatTensor)
        accuracy = matches.mean()
        test_accuracy += accuracy
        model.train()
    print(f'Model Test Accuracy: {test_accuracy/len(test_dataloader)*100:.2f}%\n')

def save_model(trained_model,hidden_units,output_units,dest_dir,model_arch,class_to_idx):
    model_checkpoint = {'model_arch':model_arch, 
                    'clf_input':25088,
                    'clf_output':output_units,
                    'clf_hidden':hidden_units,
                    'state_dict':trained_model.state_dict(),
                    'model_class_to_idx':class_to_idx,
                    }
    if dest_dir:
        torch.save(model_checkpoint,dest_dir+"/"+model_arch+"_checkpoint.pth")
        print(f"{model_arch} successfully saved to {dest_dir}")
    else:
        torch.save(model_checkpoint,model_arch+"_checkpoint.pth")
        print(f"{model_arch} successfully saved to current directory as {model_arch}_checkpoint.pth")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory', help="The location of the files used to train the model.")
    parser.add_argument('-a','--arch', help="Name of pretrained ImageNet model.",dest="arch",default="vgg16")
    parser.add_argument('-hid','--hidden_units', help="No. Hidden Units",dest="hidden_units",default=4096, type=int)
    parser.add_argument('-o','--output_units', help="No. Output Units",dest="output_units",default=102, type=int)
    parser.add_argument('-lr','--learning_rate', help="Adam Learning Rate",dest="learning_rate",default=.001, type=float)
    parser.add_argument('-e','--epochs', help="Epochs for model training",dest="epochs",default=5, type=int)
    parser.add_argument('-p','--print_every', help="Number of batches to print training metrics",dest="print_every",default=5, type=int)
    parser.add_argument('-g','--gpu', help="Use GPU (CUDA)?", action="store_true")
    parser.add_argument('-sd' ,'--save_dir', help="Set Directory destination for model checkpoint",dest="save_dir",default="")

    args = parser.parse_args()
    # Loading dataloaders from data directory
    training_files = args.data_directory
    train_dataloader,valid_dataloader,test_dataloader,class_to_idx = data_processing.load_images(training_files)
    # Loading Model
    model = args.arch
    hidden_units = args.hidden_units
    output_units = args.output_units
    dest_dir = args.save_dir
    pretrained_model = load_model.select_pretrained_model(model,hidden_units,output_units)
    # Training Model
    if args.gpu:
        device = 'cuda'
    else:
        device = 'cpu'
    lr = args.learning_rate
    epochs = args.epochs
    print_every = args.print_every
    train_model(pretrained_model,train_dataloader,valid_dataloader,test_dataloader,epochs,device,lr,print_every)
    # Save model to checkpoint
    save_model(pretrained_model,hidden_units,output_units,dest_dir,args.arch,class_to_idx)
    
if __name__ == '__main__':
    main()

