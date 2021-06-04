'''=========================================================================================
	In this case study, input given from user is one image which consist the number between
	o to 9 and our system is responsible for detecting that number.

	Currenly all the data we have in the numeric format with the range of 0 to 255 which is 
	the RGB color constant range.
	Means we provide the range of RGB constant as a input to the system and from that input
	our system is reposible to predict the particular ouput.
   
   	In this application we have separate csv files of training and testing dataset.so we 
   	dont need to use split function.
   =========================================================================================
'''

from mymodule import *

'''========================================
	Usage of function_caller()
	It takes four arguments - 
	  feature_train,target_train
	  feature_test, target_test	
	This function is responsible for -  going to call the function of classifier
	and inside those function classifier are designed in the another py file.
   ========================================
'''
def Function_caller(f_train,f_test,t_train,t_test):

	train,test = Decision_tree(f_train,f_test,t_train,t_test)
	Display_accuracy(accu_train=train,accu_test=test,str="Accuracy of Decision_tree")
	
	train,test = Random_Forest(f_train,f_test,t_train,t_test)
	Display_accuracy(accu_train=train,accu_test=test,str="Accuracy of Random_Forest")
	
	train,test = Bagging_Classifier(f_train,f_test,t_train,t_test)
	Display_accuracy(accu_train=train,accu_test=test,str="Accuracy of Bagging_Classifier")

'''===============================
	Display Accuracy function
   ===============================
'''
def Display_accuracy(accu_train,accu_test,str):
	print(str,"\n")
	print("Accuracy of training set : ",accu_train)
	print("Accuracy of testing  set : ",accu_test)
	print("-----------------------------------------------------------")

'''========================================================
	Here i take one csv file and use splitter function
	and then fit the model because due to seprate 
	train and test csv it may creates ambiguity in the
	accuracy and we cant take whole file because it has
	big records in it so.
   ========================================================
'''
def main():
	df = pd.read_csv("mnist_train.csv")

	df = df.sample(2000)

	df.rename(columns = {"5":"label"},inplace=True)

	feature = df.drop("label",axis=1)
	target  = df.label

	feature_train,feature_test,target_train,target_test=train_test_split(feature,target,test_size=0.5)

	Function_caller(feature_train,feature_test,target_train,target_test)


if __name__ == '__main__':
	main()


'''==============================================
	This main about seprates train and test
	csv fle and which take long time to build
   ==============================================
'''
'''
def main():
	training_dataset = pd.read_csv("mnist_train.csv")
	#this is huge file of record so we take only 1000 records for the
	#training , because if we take all the records then it take more
	#time to train the model

	training_dataset = training_dataset.sample(1500)
	print("shape of 1000 sample : ",training_dataset.shape)
	#print("Columns : ",training_dataset.columns)

	#1st column is label/target but in the csv the name of column is '5' so for better
	#readability we can rename the column as target.
	#and other than first columns remaning are the features.

	training_dataset.rename(columns = {"5":"label"},inplace=True)
	#print(training_dataset.columns)

	feature_train = training_dataset.drop("label",axis=1)
	target_train   = training_dataset["label"]

	print("Shape of training dataset : ")
	print("Shape of feature : {} - shape of target : {}".format(feature_train.shape,target_train.shape))

	#Now operation for testing dataset.

	testing_dataset = pd.read_csv("mnist_test.csv")

	testing_dataset = testing_dataset.sample(400)

	testing_dataset.rename(columns = {"5":"label"},inplace=True)

	feature_test = testing_dataset.drop("label",axis=1)
	target_test  = testing_dataset["label"]

	print("Shape of testing dataset : ")
	print("Shape of feature : {} - shape of target : {}".format(feature_test.shape,target_test.shape))

	print("-----------------------------------------------------------")
	Function_caller(feature_train,feature_test,target_train,target_test)
'''
