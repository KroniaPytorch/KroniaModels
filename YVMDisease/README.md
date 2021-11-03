## Yellow Mosaic Disease Detector

Problem at hand:* We strive to detect the yellow mosaic plant disease using our CNN models driven by torch and torchvision and classify the Okra(Ladyfinger) Leaf as diseased or healthy.
 - *Model Architecture:*
    1. [Click here for Raw Data](https://www.kaggle.com/manojgadde/yellow-vein-mosaic-disease)
    2. We use 4 blocks of Conv layers. Each block consists of Convolution + ReLU + Max Pooling Layers
    3. The 4 Conv Blocks are followed by 2 fully connected linear layers
    4. Used the logarithmic softmax function to get the probabilities of output being :
        - 0 is Diseased Okra Leaf
        - 1 is Fresh Okra Leaf
