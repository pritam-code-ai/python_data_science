
#Import scikit-learn dataset library
from sklearn import datasets
#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
#Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd




#Load dataset
iris = datasets.load_iris()

# Creating a DataFrame of given iris dataset.

data=pd.DataFrame({
    'sepal length':iris.data[:,0],
    'sepal width':iris.data[:,1],
    'petal length':iris.data[:,2],
    'petal width':iris.data[:,3],
    'species':iris.target
})

X=data[['sepal length', 'sepal width', 'petal length', 'petal width']]  # Features
y=data['species']  # Labels

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.70, random_state=5)

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

# prediction on test set
y_pred=clf.predict(X_test)


feature_imp = pd.Series(clf.feature_importances_,index=iris.feature_names).sort_values(ascending=False)

# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()


# Split dataset into features and labels
X=data[['petal length', 'petal width','sepal length']]  # Removed feature "sepal length"
y=data['species']
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.70, random_state=5) # 70% training and 30% test


#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

clf.fit(X_train,y_train)

# prediction on test set
y_pred=clf.predict(X_test)

n_errors = (y_pred != y_test).sum()


# Run classification metrics
print('{}: {}'.format("Random forest", n_errors))
# Model Accuracy, how often is the classifier correct?
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))







#
#
#
#
# digit recognition DATACET
#
#
#
#
#


from sklearn.datasets import load_digits

digits = load_digits()

# set up the figure
fig = plt.figure(figsize=(6, 6))  # figure size in inches
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# plot the digits: each image is 8x8 pixels
for i in range(64):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')

    # label the image with the target value
    ax.text(0, 7, str(digits.target[i]))

plt.show()

Xtrain, Xtest, ytrain, ytest = train_test_split(digits.data, digits.target,
                                                random_state=0)
model = RandomForestClassifier(n_estimators=1000)
model.fit(Xtrain, ytrain)
ypred = model.predict(Xtest)

print(classification_report(ypred, ytest))





#
#
#
#
# credit card DATACET
#
#
#
#
#









cc =  pd.read_csv("creditcard.csv")
cc = cc.sample(frac=0.06, random_state=1)



cc_train = cc.drop('Class', 1)

from sklearn.ensemble import IsolationForest
clf = IsolationForest(n_estimators=1000, max_samples=200)
#Train the model with the data.
y_value = cc['Class']
#print(y_value)
clf.fit(cc_train , y_value)

# The Anomaly scores are calclated for each observation and stored in 'scores_pred'
scores_pred = clf.decision_function(cc_train)

# scores_pred is added to the cc dataframe
cc['scores']= scores_pred
#I oberved an conflict with the name 'class'. Therefore, I have changed the name from class to category
cc = cc.rename(columns={'Class': 'Category'})



# For convinience, divide the dataframe cc based on two labels.
avg_count_0 = cc.loc[cc.Category==0]    #Data frame with normal observation

normal1 = plt.hist(avg_count_0.scores, 50,)
plt.xlabel('Score distribution of 0')
plt.ylabel('Frequency of 0')
plt.title("Distribution of isoforest score for normal observation")
plt.show()



avg_count_1 = cc.loc[cc.Category==1]    #Data frame with anomalous observation
#Plot the combined distribution of the scores

normal = plt.hist(avg_count_1.scores, 50,)
plt.xlabel('Score distribution of 1')
plt.ylabel('Frequency of 1')
plt.title("Distribution of isoforest score for anomalous observation")
plt.show()


