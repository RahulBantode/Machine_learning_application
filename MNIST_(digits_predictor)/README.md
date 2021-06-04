MNIST case study :- 
  Modified National Institute Of Standards and Technology.

Problem Statment :- MNIST Dataset are the famous dataset given by the National Institute of Standards and technology. to learn the concpet of
                    Image processing, in this case study we having the reocords of Images of digits in the pixel so according to that records
                    our model is predicts the digit for that particular dataset.
                    
Dataset Volume   :- Training Dataset :- 50,000 records  Testing Dataset :- 10,000 records
                    Shape of the whole dataset is :- (60,000 , 768) 
                    first column of the dataset is the Target/label for the reocords.
                    and rest of all the columns are the features/pixel of the digits.
                    
To fit the model following algorithms are used :-
1. DecisionTreeClassifier
2. RandomForesetClassifier
3. BaggingClassifier

Conclusion :-  RandomForestClassifier and BaggingClassiifer gives the better result as compare to the DecisionTreeClassifier.
               Because the DecisionTreeClassiifer comes under the singleton algorithm technique,which are fitted once and gives and
               accurarcy and basically it goes to the overfitting problem where (bias is low and variance is high).
               
               RandomForest and BaggingClassifier comes under the Ensemble algorithm technique which best fits the model. BaggingClassifier
               are the advance version of RandomForsetClassifier in RandomForestClassifier base_estimator by default DecisionTreeClassifier
               we dont have choice to change it, but in baggingClassifier we can change the base_esitmator with any of the algorithm apart from
               DecisionTreeClassifier,or incase we failed to pass the base_estimator, then it will consider the DecisionTree as a base_esitmator,whcih
               is default value for the BaggingClassifier.
