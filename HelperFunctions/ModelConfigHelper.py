import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

from torchvision import models

import matplotlib.pyplot as plt

def getPreTrainedModel(model_name,preTrained=True):
    if model_name == 'alexnet':
        model = models.alexnet(pretrained=preTrained)
        return model
    elif model_name == 'vgg19':
        model = models.vgg19_bn(pretrained=preTrained)
        return model
    elif model_name == 'inception':
        model = models.inception_v3(pretrained=preTrained)
        return model
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=preTrained)
        return model
    else:
        print("Please enter a valid model")
        return

def set_grad(model,feature_extract=False):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False

def getCustomizedPreTrainedModel(model_name,final_node_count,feature_extract,preTrained=True):
    if model_name == 'alexnet':
        model = models.alexnet(pretrained=preTrained)
        set_grad(model,feature_extract=feature_extract)
        num_in = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_in,final_node_count)
        return model
    elif model_name == 'vgg19':
        model = models.vgg19_bn(pretrained=preTrained)
        set_grad(model,feature_extract=feature_extract)
        num_in = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_in,final_node_count)
        return model
    elif model_name == 'inception':
        model = models.inception_v3(pretrained=preTrained)
        set_grad(model,feature_extract=feature_extract)
        num_in = model.fc.in_features
        model.fc = nn.Linear(num_in,final_node_count)
        return model
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=preTrained)
        set_grad(model,feature_extract=feature_extract)
        num_in = model.fc.in_features
        model.fc = nn.Linear(num_in,final_node_count)
        return model
    else:
        print("Please enter a valid model")
        return

def trainModel(model,train_dl,val_dl,criterion,optim,train_samples,batch_size=32,soft_max=False,epochs=5):
    start_time = time.time()
    train_losses = []
    test_losses = []
    train_correct = []
    test_correct = []
    for i in range(epochs):
        trn_corr = 0
        tst_corr = 0
        print("============= New Epoch =========================")
        # Run the training batches
        for b, (X_train, y_train) in enumerate(train_dl):
            b+=1
            # Apply the model
            y_pred = model(X_train)  # we don't flatten X-train here
            if soft_max == True:
                y_pred = F.log_softmax(y_pred,dim=1)
            y_train = y_train.type(torch.LongTensor)
            loss = criterion(y_pred, y_train)
 
            # Tally the number of correct predictions
            predicted = torch.max(y_pred.data, 1)[1]
            batch_corr = (predicted == y_train).sum()
            trn_corr += batch_corr
        
            # Update parameters
            optim.zero_grad()
            loss.backward()
            optim.step()
        
            # Print interim results
            sample_size = min(batch_size*b,train_samples)
            if b%2==0:
                print(f'epoch: {i:2}  batch: {b:4} [{sample_size:6}/{train_samples}]  loss: {loss.item():10.8f}  \
    accuracy: {trn_corr.item()*100/(sample_size):7.3f}%')
        
        train_losses.append(loss)
        train_correct.append(trn_corr)
        
        # Run the testing batches
        with torch.no_grad():
            for b, (X_test, y_test) in enumerate(val_dl):
                # Apply the model
                y_val = model(X_test)
                if soft_max == True :
                    y_val = F.log_softmax(y_val,dim=1)
                # Tally the number of correct predictions
                predicted = torch.max(y_val.data, 1)[1] 
                tst_corr += (predicted == y_test).sum()
    
        y_test = y_test.type(torch.LongTensor)      
        loss = criterion(y_val, y_test)
        test_losses.append(loss)
        test_correct.append(tst_corr)
        
    print(f'\nDuration: {time.time() - start_time:.0f} seconds') # print the time elapsed
    return train_losses,test_losses

def visualiziseTrainResults(train_losses,test_losses):
    t = [x.detach().numpy() for x in train_losses]
    plt.plot(t, label='training loss')
    plt.plot(test_losses, label='validation loss')
    plt.title('Loss at the end of each epoch')
    plt.legend();

def saveModel(model,filename):
    torch.save(model.state_dict(),filename+'.pth')