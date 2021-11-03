## Rice Disease Classification
*Problem at hand:* We strive to detect the rice plant diseases using our CNN models driven by torch and torchvision and classify the same.
 - *Model Architecture:*
    1. [Click here for Raw Data](https://archive.ics.uci.edu/ml/datasets/Rice+Leaf+Diseases)
    2. We use a pre-trained AlexNet model(feature-extraction) with customizations in the final layer to make predictions on the dataset
    3. Using the Logarithmic Softmax function we get the probabilities of output being : 
       - 0 is Bacterial Blight 
       - 1 is Blast
       - 2 is Brownspot
       - 3 is Tungro
