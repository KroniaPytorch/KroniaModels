## Crop Recommender

*Problem at hand:* We strive to advise the best crops to the farmers and research-based agriculturists based on their soil and fertilizer attributes including NPK and ph level of the soil using our ANN models driven by PyTorch.
 - *Model Architecture:*
   1. [Click here for Raw Data](https://www.kaggle.com/atharvaingle/crop-recommendation-dataset)
   2. Defined a 3-layer feed-forward network with dropout and batch-norm.
   3. Used the nn.CrossEntropyLoss because this is a multiclass classification problem.
   4. No requirement of log_softmax layer after our final layer because nn.CrossEntropyLoss does the job
