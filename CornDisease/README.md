## Corn Disease Detection

*Problem at hand:* We strive to detect the corn plant diseases using our CNN models driven by torch and torchvision and classify the same as infected or healthy corn.
  - *Model Architecture:*
    1. [Click here for Raw Data](https://www.kaggle.com/qramkrishna/corn-leaf-infection-dataset) or you can find a resized version of the dataset [here](https://drive.google.com/file/d/1VX-HhhU6uzY_CgXKKKUeH0Mw5GUfRTC_/view?usp=sharing)
    2. We use a pre-trained AlexNet model(feature-extraction) with customizations in the final layer to make predictions on the dataset
    3. Using a logarithmic softmax layer we get the probabilities of output being 0 or 1 where 0 is healthy and 1 is infected
       
