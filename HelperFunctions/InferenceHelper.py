from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

def saveModel(model,filename):
    torch.save(model.state_dict(),filename+'.pth')

def getTestImg(path):
    img = Image.open(path)
    return img

def preProcess(img,transform):
    img = transform(img)
    img = torch.unsqueeze(img,dim=0)
    return img

def inference(img_path,model,weight_path,dictlabel,transform,softmax=False):
    img = getTestImg(img_path)
    img = preProcess(img,transform)
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    with torch.no_grad():
        y = model(img)
    if softmax == False :
        y = F.log_softmax(y,dim=1)
    predicted = torch.max(y.data,1)[1]
    return dictlabel[predicted]