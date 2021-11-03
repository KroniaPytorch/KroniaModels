# Kronia Models
This repository contains all the Pytorch models and the related code used in the development of this project

## Motivation
Our prime motivation for making this App was to provide an application that can be used by farmers and agricultural workers all around the globe. We wanted to make sure that our application provides a ton of really helpful features and also delivers it with a certain level of finesse.

## Repository Overview

## Tech Stack

The following libraries for the development of our p :

- Pytorch (torch and torchvision)
- Sklearn 
- Pillow
- Numpy
- Pandas
- Matplotlib

## Related Repos
The list of related repositories that were used to develop Kronia can be found here

## Setup and Install
To properly setup this project on a local machine , please carefully go through the following steps

### Install Anaconda

-
### Create a new environment

- We highly recommend the readers to create a new virtual environment for working on this project,as this will prevent users from messing up their local environments (that have already been setup)
- For beginners and new users,please refer this link to know more about virtual environments and how to setup a new virtual environment
- To setup a new environment :
    - Beginner Friendly 
        - If you haven't installed Anaconda please install Anaconda and then Open the Anaconda Navigator.
        - Click on the environment tab 
        - Click on the create button to create an environment
    - Advanced
       -   Run ``` conda create name_of_env ```
       -   Run ``` activate name_of_env ``` (Windows) or ```source activate name_of_env``` (Linux/MacOS) to switch into the created environment.(More conda cheat sheets can be found here)

### Install the dependencies

- Now we need to install all the libraries that have been listed in the tech stack section.
  - Install Pytorch
  - Install Sklearn
  - Install Pillow
  - Install Numpy
  - Install Pandas
  - Install Matplotlib
### Clone this repository

- Move into the desired location on your local setup and make sure that you are using the correct virtual environment.
- In the appropriate termial, run ``` git clone https://github.com/KroniaPytorch/KroniaModels.git``` to get the code base
### Inference

- Once the dependencies have been installed and the code-base cloned,the user can now train the model or test in on custom images
- Please keep in mind the folder heirarchy and move into the appropriate folder
- To train the model again (with your tweaked settings) ,run the notebook with the name that matches the folder
- To test the model , make sure you have downloaded the weights(read next step) or have generated a new weight file by training the model and then run Inference notebook in the folder
- Weights for the particular model can be downloaded from here
- Please note : Before Running the model, make sure you are running the notebook from the appropriate environment
### Quick Setup

- If for some reason the user finds some trouble in installing the code , we also provide "Quick-Use" files that can be used by the user to test the models on cus
- The user can find the files here
- The user will find an Inference ("Quick Use") file that can be executed in Google Colab along with the appropriate weight files(available in the link mentioned above)
