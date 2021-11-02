import sys
import os
import pandas as pd

def getFileDirectory(fetchType='folder'):
    path = input("Please enter the root directory of the file path -> ")
    if (fetchType == 'folder'):
        assert os.path.isdir(path) , "Couldn't find the "+fetchType+" at the path provided : " + str(path)
    elif (fetchType == 'file'):
        assert os.path.isfile(path) , "Couldn't find the "+fetchType+" at the path provided : " + str(path)
    return {'path':path,'type':fetchType}

def showFileCount(pathInfo):
    if isinstance(pathInfo,dict):
        if ('type' in pathInfo) and ('path' in pathInfo):
            if pathInfo['type'] == 'folder' :
                for fol in os.listdir(pathInfo['path']):
                    files = [file for file in os.listdir(os.path.join(pathInfo['path'],fol))]
                    print("============================================")
                    print (fol+" :-")
                    print(len(files))
            else:
                print("Wrong path type")
        else:
            print("The input did not have the appropriate keys")
    else:
        print("Please enter a valid input")

def getDataFrame(pathInfo,normalImgKey='normal'):
    if isinstance(pathInfo,dict):
        if ('type' in pathInfo) and ('path' in pathInfo):
            if pathInfo['type'] == 'folder' :
                data={'imgPath':[],'label':[]}
                for fol in os.listdir(pathInfo['path']):
                    for file in os.listdir(os.path.join(pathInfo['path'],fol)):
                        data['imgPath'].append(os.path.join(pathInfo['path'],fol,file))
                        data['label'].append(fol)
                        
                df = pd.DataFrame(data)
                df['augmentation'] = normalImgKey
                df = df.sample(frac=1).reset_index(drop=True)
                df['label'] = pd.Categorical(df['label'])
                return df
            else:
                print("Wrong path type")
        else:
            print("The input did not have the appropriate keys")
    else:
        print("Please enter a valid input")

def getLabelDicts(df):
    if isinstance(df,pd.DataFrame):
        dictlabel = df['label'].cat.categories
        return dictlabel
    else:
        print('Entered input is not a pandas dataframe')

def dfPreProcess(df):
    if isinstance(df,pd.DataFrame):
        df['ylabel'] = df['label'].cat.codes
        df.drop('label',axis=1,inplace=True)
        return df
    else:
        print('Entered input is not a pandas dataframe')
