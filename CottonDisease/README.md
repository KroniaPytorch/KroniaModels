# Cotton Plant Disease Classifier 
*Problem at hand:* We strive to detect the cotton plant diseases using our CNN models driven by torch and torchvision and classify the same.
  - *Model Architecture:*
     1. [Click here for Raw Data](https://www.kaggle.com/raaavan/cottonleafinfection)
     2. We use 5 blocks of Conv layers. Each block consists of Convolution + ReLU + MaxPool layers.
     3. The Conv layers are followed by 2 fully connected linear layer
     4. Used the Logarithmic Softmax function to get the probabilities of output being : 
        - 0 is Bacterial Blight 
        - 1 is Curl Virus
        - 2 is Fussarium Wilt
        - 3 is Healthy
