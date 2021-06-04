'''===========================================================================
	The Diabetic case study - solved by following four algorithms
		1. DecisionTreeClassifer
		2. KNeighborsClassifier
		3. LogisticRegression
		4. RandomForestClassifier
	In this application we are not doing any cleaning in further application 
	make some cleaning.
   ===========================================================================
'''

from mymodule import *

def Function_caller(f_train,f_test,t_train,t_test):

	train,test = Decision_tree(f_train,f_test,t_train,t_test)

	Display_accuracy(accu_train=train,accu_test=test,str="Accuracy of DT")

	train,test = K_nearest_neighbor(f_train,f_test,t_train,t_test)
	Display_accuracy(accu_train=train,accu_test=test,str="Accuracy of KNN")

	train,test = Logistic_regression(f_train,f_test,t_train,t_test)
	Display_accuracy(accu_train=train,accu_test=test,str="Accuracy of LG")

	train,test = Random_Forest(f_train,f_test,t_train,t_test)
	Display_accuracy(accu_train=train,accu_test=test,str="Accuracy of RF")
	
	

def Display_accuracy(accu_train,accu_test,str):
	print(str)
	print("Accuracy of training set : ",accu_train)
	print("Accuracy of testing  set : ",accu_test)
	print("-----------------------------------------------------------")


def main():
	dataset = pd.read_csv("Diabetes.csv")

	feature = dataset.drop("Outcome",axis = 1)
	target  = dataset["Outcome"]

	feature_train,feature_test,target_train,target_test = train_test_split(feature,target,test_size=0.5)

	Function_caller(feature_train,feature_test,target_train,target_test)


if __name__ == '__main__':
	main()