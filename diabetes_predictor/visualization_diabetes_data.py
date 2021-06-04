'''=============================================================
	Code consist the visualization over the dataset of diabetes
   =============================================================
'''
from mymodule import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def main():
	dataset = pd.read_csv("diabetes.csv")
	print("Shape of dataset : ",dataset.shape)

	'''=============================================================================
		Visualization of How many people has diabetes or how many not in the form of
		bar graph.
		0 = no diabetes
		1 = yes diabetes
	   =============================================================================
	'''
	figure()
	print(dataset.Outcome.value_counts()) #this function gives the count of 0 and 1.
	countplot(data=dataset,x="Outcome").set_title("diabetes vs no-diabetes")
	#show()

	'''=============================================================================
		Visualization according to number of pregnacies (is affect on diabetes or not)
	   =============================================================================
	'''
	figure()
	#countplot(data=dataset,x="Outcome",hue="Pregnancies").set_title("Visualization acc to pregnacies column")
	dataset["Pregnancies"].plot.hist().set_title("Visualization acc to pregnacies")
	#show()

	'''=============================================================================
		Visualization according to glucose.
	   =============================================================================
	'''
	figure()
	#countplot(data=dataset,x="Outcome",hue="Glucose").set_title("Visualization acc to glucose column")
	dataset["Glucose"].plot.hist().set_title("Visualization acc to glucose")
	#show()


	'''=============================================================================
		Visualization according to blood pressure
	   =============================================================================
	'''
	figure()
	#countplot(data=dataset,x="Outcome",hue="BloodPressure").set_title("Visualization acc to BloodPressure column")
	#show()
	#dataset["BloodPressure"].plot.hist().set_title("Visualization acc to BloodPressure")
	#show()

	'''=============================================================================
		Visualization according to skin thickness
	   =============================================================================
	'''
	dataset["SkinThickness"].plot.hist().set_title("Visualization acc to SkinThickness")
	show()

	X_feature = dataset.drop("Outcome",axis=1)
	Y_target  = dataset["Outcome"]

	'''
	tree = DecisionTreeClassifier()

	tree.fit(X_feature,Y_target)

	plot_feature_importances_diabetes(tree,dataset)
	'''
	
	rf = RandomForestClassifier()
	rf.fit(X_feature,Y_target)
	plot_feature_importances_diabetes(rf,dataset)
	
'''=============================================================
	plot the graph of which features are important in dataset
	=============================================================
'''
def plot_feature_importances_diabetes(model,dataset):
		
	plt.figure(figsize=(9,8)) #first para is width and second is height of the figure
	n_features = 8  #ther is 8 features and 1 target column in the dataset

	#feature_importance function associated with model like DT,KNN,LG etc.
	plt.barh(range(n_features),model.feature_importances_,align="center") 
		
	#following codes gives us the list of feature column
	#enumerate :- when we use this function in our looping it return the tuple of value and the
	#counter you dont need to implement or increment counter explicitly.
	diabetes_feature = [x for i, x in enumerate(dataset.columns) if i!=8]
	plt.yticks(np.arange(n_features),diabetes_feature)
	plt.xlabel("Feature Importance")
	plt.ylabel("Features")
	plt.ylim(-1,n_features)
	plt.show()

	
if __name__ == '__main__':
	main()