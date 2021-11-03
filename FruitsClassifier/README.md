## Fruit Classifier

*Problem at hand:* We strive to detect the fruit plants from uncharted and exotic orchards using our CNN models driven by torch and torchvision and classify the same.
 - *Model Architecture:*
   1. [Click here for Raw Data](https://www.kaggle.com/vaishnavikhilari/fruits-recognition)
   2. We use 3 blocks of Conv layers. Each block consists of Convolution + ReLU + Max Pooling Layer.
   3. The 3 Conv blocks are followed by 2 fully connected layers
   4. Used the Logarithmic Softmax function to classify among several fruits ranging from Apples to Huckleberries
