'''===============================================================================
	Diabetes predictor :-
	In this case study we fill all the zeros from the columns by their respectives
	column mean and then check whether getting good accuracy or not.
   	
   	and one more classifier we used in this application name as - BaggingClassifier()
   ================================================================================
'''
from mymodule import *

def preprocess_dataset(df):
	''' visualize the zeros in the columns
	    row = (df.loc[:,"Insulin"] == 0)
	    print(row.value_counts())

		by doing this code in IDLE directly on the column of Insulin and SkinThichness
		which gives large number of zeros as comapare to the blood pressure.

		so now we have to replace all the zeros from the Insulin and SkinThickness by mean of
		that columns
	'''
	mean_insulin = df["Insulin"].mean()
	mean_skinthickness = df["SkinThickness"].mean()

	'''===============================================================
		finding of zeros rows from insulin,skinthickness columns
		and update all the zeros with mean of that respective columns
	   ===============================================================
	'''
	zeros_insulin = (df.loc[:,"Insulin"] == 0)
	df.loc[zeros_insulin,"Insulin"] = mean_insulin

	zeros_skinthickness = (df.loc[:,"SkinThickness"] == 0)
	df.loc[zeros_skinthickness,"SkinThickness"] = mean_skinthickness

	return df

'''==============================================
	Function calls the all the functions from 
	ML_Algorithms.py file where all the function
	contains respectives classifier
   ==============================================
'''
def Function_caller(f_train,f_test,t_train,t_test):

	train,test = Decision_tree(f_train,f_test,t_train,t_test)

	Display_accuracy(accu_train=train,accu_test=test,str="Accuracy of DT")

	train,test = K_nearest_neighbor(f_train,f_test,t_train,t_test)
	Display_accuracy(accu_train=train,accu_test=test,str="Accuracy of KNN")

	train,test = Logistic_regression(f_train,f_test,t_train,t_test)
	Display_accuracy(accu_train=train,accu_test=test,str="Accuracy of LG")

	train,test = Random_Forest(f_train,f_test,t_train,t_test)
	Display_accuracy(accu_train=train,accu_test=test,str="Accuracy of RF")
	
	train,test = Bagging_Classifier(f_train,f_test,t_train,t_test)
	Display_accuracy(accu_train=train,accu_test=test,str="Accuracy of Bagging_Classifier")

	
'''================================
	Displays the accuracy
   ================================
'''
def Display_accuracy(accu_train,accu_test,str):
	print(str)
	print("Accuracy of training set : ",accu_train)
	print("Accuracy of testing  set : ",accu_test)
	print("-----------------------------------------------------------")


def main():
	dataset = pd.read_csv("diabetes.csv")

	#now this is updated dataset
	dataset = preprocess_dataset(dataset)	
	#print(dataset["Insulin"].head(10))

	feature = dataset.drop("Outcome",axis=1)
	target  = dataset["Outcome"]

	feature_train,feature_test,target_train,target_test = train_test_split(feature,target,test_size=0.5)

	Function_caller(feature_train,feature_test,target_train,target_test)

if __name__ == '__main__':
	main()