## SoilNet 

*Problem at hand:* We strive to detect the soil outlier and filter the same before the image is run on the native Soil classifier API using our CNN models driven by torch and torchvision.
 - *Model Architecture:*
   1. [Click here for Raw Data](https://www.kaggle.com/qramkrishna/corn-leaf-infection-dataset)
   2. We use a pre-trained AlexNet model(feature-extraction) with customizations in the final layer to make predictions on the dataset
   3. Using a logarithmic softmax layer we get the probabilities of output being : 
      - 0, where 0 is Non Soil Images
      - 1, where 1 is Soil Images
   
