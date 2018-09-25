# MachineLearning

Problem Statement: The skLearn package in Python does not support stepwise selection.In this code we will add a stepwise selection component to it.
                   
Dataset: Used the 'compustat_annual_2000_2017_with link information' as the dataset.

Python file: Stepwise_Linear_Regression

Code: 'StepwiseR' function runs skLearn linear_model iteratively, and at each iteration goes through stepwise process to choose the   variables.Used a level of significance of 0.05 for critical value. Finally used the linear regression to predict operational profit. 
       
