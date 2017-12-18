# SVM-demo

```python
# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('kaggle_titanic_train.csv')
X = pd.DataFrame([dataset["Pclass"],
                  dataset["Sex"],
                  dataset["Age"]]).T
y = dataset[["Survived"]]

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
X["Age"] = imputer.fit_transform(X["Age"].reshape(-1, 1))

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X["Sex"] = labelencoder.fit_transform(X["Sex"])

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting SVM to the Training set
from sklearn.svm import SVC
clf = SVC(C=1, gamma=0.1)
clf.fit(X_train, y_train)

# Predicting the Test set results
y_pred = clf.predict(X_test)

# Making the Confusion Matrix and calculating accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# Fitting SVM to the Training set
clf2 = SVC(C=2, gamma=0.2)
clf2.fit(X_train, y_train)

# Predicting the Test set results
y_pred2 = clf2.predict(X_test)

# Making the Confusion Matrix and calculating accuracy
cm2 = confusion_matrix(y_test, y_pred2)
print(cm2)
accuracy2 = accuracy_score(y_test, y_pred2)
print(accuracy2)
```

# K means-demo

```python
# Importing the libraries
import matplotlib.pyplot as plt

# Importing the dataset
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)

# Fitting K-Means to the dataset
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 3, random_state = 0)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], c = 'red')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], c = 'blue')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], c = 'green')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'yellow')
```
