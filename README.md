# Kronia Models
This repository contains all the Pytorch models and the related code used in the development of this project

## Motivation
Our prime motivation for making this App was to provide an application that can be used by farmers and agricultural workers all around the globe. We wanted to make sure that our application provides a ton of really helpful features and also delivers it with a certain level of finesse.

## Repository Overview

## Tech Stack

The following libraries for the development of our project :

- [Pytorch](https://pytorch.org/) (torch and torchvision)
- [Sklearn](https://scikit-learn.org/stable/) 
- [Pillow](https://pypi.org/project/Pillow/)
- [Numpy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)

## Related Repos
The list of related repositories that were used to develop Kronia can be found [here](https://github.com/KroniaPytorch)

## Setup and Install
To properly set up this project on a local machine, please carefully go through the following steps

### Install Anaconda

- Users can install Anaconda from [here](https://www.anaconda.com/products/individual)
### Create a new environment

- We highly recommend the readers create a new virtual environment for working on this project, as  this will prevent users from messing up their local environments (that have already been setup)
- For beginners and new users, please refer to this [link](https://www.datacamp.com/community/tutorials/virtual-environment-in-python) to know more about virtual environments and how to set up a new virtual environment
- To set up a new environment :
    - Beginner Friendly 
        - If you haven't installed Anaconda please install Anaconda and then Open the Anaconda Navigator.
        - Click on the environment tab (visible on the left side of the window)
        - Click on the create button to create an environment
    - Advanced
       -   Run ``` conda create name_of_env ```
       -   Run ``` activate name_of_env ``` (Windows) or ```source activate name_of_env``` (Linux/macOS) to switch into the created environment.(An anaconda cheatsheet can be found [here](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwj0oJzz0PzzAhUA7HMBHY8BBQ4QFnoECAwQAQ&url=https%3A%2F%2Fdocs.conda.io%2Fprojects%2Fconda%2Fen%2F4.6.0%2F_downloads%2F52a95608c49671267e40c689e0bc00ca%2Fconda-cheatsheet.pdf&usg=AOvVaw3uUYEqas7NMuAmCCWAx_yl))

### Install the dependencies

- Now we need to install all the libraries that have been listed in the tech stack section.
  - Install [Pytorch](https://pytorch.org/) 
  - Install [Sklearn](https://scikit-learn.org/stable/install.html)  or Install from Anaconda Navigator
  - Install [Pillow](https://anaconda.org/anaconda/pillow) or Install from Anaconda Navigator
  - Install [Numpy](https://anaconda.org/anaconda/numpy) or Install from Anaconda Navigator
  - Install [Pandas](https://pandas.pydata.org/docs/getting_started/install.html#installing-with-anaconda) or Install from Anaconda Navigator
  - Install [Matplotlib](https://anaconda.org/conda-forge/matplotlib) or Install from Anaconda Navigator
### Clone this repository

- Move into the desired location on your local setup and make sure that you are using the correct virtual environment.
- In the appropriate terminal, run ``` git clone https://github.com/KroniaPytorch/KroniaModels.git``` to get the codebase### Inference

- Once the dependencies have been installed and the code-base cloned, the user can now train the model or test it on custom images
- Please keep in mind the folder hierarchy and move it into the appropriate folder
- To train the model again (with your tweaked settings), run the notebook with the name that matches the folder
- To test the model, make sure you have downloaded the weights(read next step) or have generated a new weight file by training the model and then run Inference notebook in the folder
- Weights for the particular model can be downloaded from [here](https://drive.google.com/drive/folders/1UXVMipuO_Dvskdv2g_4cMqoLIKRzi1HZ?usp=sharing)
- Please note: Before Running the model, make sure you are running the notebook from the appropriate environment
### Quick Setup

- If for some reason the user finds some trouble in installing the code, we also provide "Quick-Use" files that can be used by the user to test the models on cus
- The user can find the files [here](https://drive.google.com/drive/folders/1UXVMipuO_Dvskdv2g_4cMqoLIKRzi1HZ?usp=sharing)
- The user will find an Inference ("Quick Use") file that can be executed in Google Colab along with the appropriate weight files(available in the link mentioned above)
