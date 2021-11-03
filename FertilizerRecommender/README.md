## Fertilizer Recommender

*Problem at hand:* We strive to advise the best fertilizers and quick buy portal to the farmers and research-based agriculturists based on their soil NPK values and climate conditions in the area using our ANN models driven by PyTorch.
  - *Model Architecture:*
     1. [Click here for Raw Data](https://www.kaggle.com/gdabhishek/fertilizer-prediction?select=Fertilizer+Prediction.csv)
     2. Defined a 3-layer feed-forward network with dropout and batch-norm.
     3. Used the nn.CrossEntropyLoss because this is a multiclass classification problem.
     4. No requirement of log_softmax layer after our final layer because nn.CrossEntropyLoss does the job.
