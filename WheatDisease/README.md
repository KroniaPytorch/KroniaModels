## Wheat Disease Classification

*Problem at hand:* We strive to detect the wheat plant diseases using our CNN models driven by torch and torchvision and classify the same.
 - *Model Architecture:*
   1. [Click here for Raw Data](https://www.kaggle.com/shadabhussain/cgiar-computer-vision-for-crop-disease)
   2. We use 4 blocks of Conv layers. Each block consists of Convolution + BatchNorm + ReLU + Dropout layers.
   3. Used the nn.CrossEntropyLoss to get the probabilities of output being 0, 1 or 2 where 0 is healthy, 1 is leaf rust and 2 is stem rust.
   4. No requirement of log_softmax layer after our final layer because nn.CrossEntropyLoss does the job.
