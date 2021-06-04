'''=======================================================
	This file contains the function with respective their
	classifier(algorithms)
   ========================================================
'''

from mymodule import *

'''==========================
	DecisionTreeClassifier
   ==========================
'''
def Decision_tree(f_train,f_test,t_train,t_test):

	#obj = DecisionTreeClassifier(max_depth=5)
	obj = DecisionTreeClassifier()

	obj.fit(f_train,t_train)

	training_accuracy = obj.score(f_train,t_train)*100

	#output = obj.predict(f_test)

	testing_accuracy = obj.score(f_test,t_test)*100

	return training_accuracy,testing_accuracy


'''======================================
	RandomForestClassifier
   ======================================
'''
def Random_Forest(f_train,f_test,t_train,t_test):
	
	obj = RandomForestClassifier(n_estimators=30,random_state=3)

	obj.fit(f_train,t_train)

	training_accuracy = obj.score(f_train,t_train)*100

	#output = obj.predict(f_test)

	testing_accuracy = obj.score(f_test,t_test)*100

	return training_accuracy,testing_accuracy


'''==========================
	BaggingClassifier	
   ==========================
'''
def Bagging_Classifier(f_train,f_test,t_train,t_test):
	
	#DecisionTreeClassifier()
	obj = BaggingClassifier(base_estimator=RandomForestClassifier(n_estimators=10),n_estimators=20)
	
	obj.fit(f_train,t_train)

	training_accuracy = obj.score(f_train,t_train)*100

	#output = obj.predict(f_test)

	testing_accuracy = obj.score(f_test,t_test)*100

	return training_accuracy,testing_accuracy
