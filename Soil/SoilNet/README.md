## SoilNet 

*Problem at hand:* We strive to detect the soil outlier and filter the same before the image is run on the native Soil classifier API using our CNN models driven by torch and torchvision.
 - *Model Architecture:*
   1. [Click here for Raw Data](https://drive.google.com/drive/folders/1qTjWZ8kupb7UrSLdmy1qJA9XMOu4RTWf?usp=sharing)
   2. We use a pre-trained AlexNet model(feature-extraction) with customizations in the final layer to make predictions on the dataset
   3. Using a logarithmic softmax layer we get the probabilities of output being : 
      - 0, where 0 is Non Soil Images
      - 1, where 1 is Soil Images
   
