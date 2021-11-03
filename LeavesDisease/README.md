## Plant Leaf Disease Classifier 

*Problem at hand:* We strive to detect the plant leaf diseases using our CNN models driven by torch and torchvision and classify the same into a wide array of disease classes.
 - *Model Architecture:*
   1. [Click here for Raw Data](https://www.kaggle.com/vipoooool/new-plant-diseases-dataset)
   2. We use 4 blocks of Conv layers and 2 Residual blocks to create a "Mini ResNet" model. Each conv block consists of Convolution + BatchNorm + ReLU
   3. The above layers are followed by a Fully Connected Linear layer
   4. Used the Logarithmetic Softmax function to predict diseases ranging from Potato's early blights to scabs in Apples
