# ml_playground 

1. tic tac toe ai
   = implementation of q-learning
   
   features
   - ai learns from experience and gets better over time
   - play x vs o, ai vs ai, or human vs ai
   
   tech stack
   - python 3
   - jupyter notebook

3. placement prediction
   = implementation of linear regression

   features
   - visualises data using matplotlib
   - splits data into training and testing sets
   - scales features for better performance
   - trains a logistic regression model using scikit-learn
   - plots decision boundaries to show model behaviour

   tech stack
   - python 3
   - jupyter notebook
   - libraries used = numpy, pandas, matplotlib, scikit-learn, mlxtend 

4. mnist neural net
   = complete lifecycle of a neural network from training to inference

   features
   - custom feed forward neural network built using torch.nn
   - trains on mnist dataset (handwritten digits = 0 to 9)
   - implements full training loop (forward pass, loss calculation, backpropagation, optimization)
   - saves and reloads the model
   - performs inference on unseen data from the test data set
   - cpu/ gpu auto-detection for flexible training

   tech stack
   - language = python
   - deep learning framework = pytorch
   - dataset = mnist
   - model type = feedforward (fully connected) neural network
   - environment = works on cpu/ gpu (cuda if available)
   - libraries used = torchvision (visualization), numpy (data handling)  
