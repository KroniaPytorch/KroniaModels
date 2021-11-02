import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

def getAugmentation(augmentRequired=False,augmentPolicies=[],angle=90):
    augDict={'hFlip':transforms.RandomHorizontalFlip(p=1),'vFlip':transforms.RandomVerticalFlip(p=1),'rot':transforms.RandomRotation(angle)}
    if augmentRequired == False :
        return augDict
    else :
        if len(augmentPolicies) == 0 :
            return augDict
        else:
            for i in range(len(augmentPolicies)):
                if isinstance(augmentPolicies[i],transforms.AutoAugmentPolicy):
                    augDict['aug'+str(i)] = transforms.AutoAugment(augmentPolicies[i])
            return augDict

def augmentDataFrame(df,augDict):
    data={'imgPath':[],'ylabel':[],'augmentation':[]}
    l=list(augDict.keys())
    for i in range(df.shape[0]):
        for j in range(len(l)):
            data['imgPath'].append(df.iloc[i,0])
            data['ylabel'].append(df.iloc[i,2])
            data['augmentation'].append(l[j])
    df2 = pd.DataFrame(data)
    dfs = [df,df2]
    DF = pd.concat(dfs)
    DF = DF.sample(frac=1).reset_index(drop=True)
    return DF

def getImageTransform(box_dim=227):
    imgTransforms = transforms.Compose([
        transforms.Resize((box_dim,box_dim)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ])
    return imgTransforms

class KroniaDataset(Dataset):
    def __init__(self,data,augDict=None,transforms=None):
        self.imgs = data
        self.transforms = transforms
        self.augs = augDict
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        aug = self.imgs.iloc[idx,1]
        image = Image.open(self.imgs.iloc[idx,0]).convert('RGB')
        label = self.imgs.iloc[idx,2]
        if aug != 'normal':
            if self.augs:
                image = self.augs[aug](image)
        if self.transforms:
            image = self.transforms(image)
        return image,label
