## Soil Classifier

*Problem at hand:* We strive to detect the soils present in the farm fields and extract significant attributes like the NPK value and ph level found in the classified soil using our CNN models driven by torch and torchvision.
 - *Model Architecture:*
    1. [Click here for Raw Data](https://drive.google.com/drive/folders/131exKxt_NAUfHBJZLnWYWZMiHSJYGSHy?usp=sharing)
    2. We use 5 blocks of Conv layers. Each block consists of Convolution +  ReLU + Max Pooling layers.
    3. Used the Logarithmic Softmax Function to get the probabilities of output being : 
        - 0 is Black Soil
        - 1 is Clayey Soil
        - 2 is Loamy Soil
        - 3 is Red Soil
        - 4 is Sandy Soil
