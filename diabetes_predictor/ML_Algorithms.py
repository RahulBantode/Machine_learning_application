from mymodule import *

'''==========================
	DecisionTreeClassifier
   ==========================
'''
def Decision_tree(f_train,f_test,t_train,t_test):

	#obj = DecisionTreeClassifier(max_depth=3)
	obj = DecisionTreeClassifier()

	obj.fit(f_train,t_train)

	training_accuracy = obj.score(f_train,t_train)*100

	#output = obj.predict(f_test)

	testing_accuracy = obj.score(f_test,t_test)*100

	return training_accuracy,testing_accuracy

'''==========================
	KNeighborsClassifier	
   ==========================
'''
def K_nearest_neighbor(f_train,f_test,t_train,t_test):

	#obj = KNeighborsClassifier()
	obj = KNeighborsClassifier(n_neighbors=7)

	obj.fit(f_train,t_train)

	training_accuracy = obj.score(f_train,t_train)*100

	#output = obj.predict(f_test)

	testing_accuracy = obj.score(f_test,t_test)*100

	return training_accuracy,testing_accuracy

'''==========================
	LogisticRegression	
   ==========================
'''
def Logistic_regression(f_train,f_test,t_train,t_test):

	obj = LogisticRegression(max_iter=500)

	obj.fit(f_train,t_train)

	#score method internally calls the predict method- so we dont need to call it exceplicitly.
	training_accuracy = obj.score(f_train,t_train)*100

	#output = obj.predict(f_test)

	testing_accuracy = obj.score(f_test,t_test)*100

	return training_accuracy,testing_accuracy


'''==========================
	RandomForestClassifier	
   ==========================
'''
def Random_Forest(f_train,f_test,t_train,t_test):
	
	obj = RandomForestClassifier(n_estimators=20)

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
	obj = BaggingClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=20)
	
	obj.fit(f_train,t_train)

	training_accuracy = obj.score(f_train,t_train)*100

	#output = obj.predict(f_test)

	testing_accuracy = obj.score(f_test,t_test)*100

	return training_accuracy,testing_accuracy
