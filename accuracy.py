from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
iris=load_iris()
knn=KNeighborsClassifier(n_neighbors=5)
x=iris.data
y=iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
knn.fit(x_train,y_train)
y_predict=knn.predict(x_test)
print(accuracy_score(y_test,y_predict))
