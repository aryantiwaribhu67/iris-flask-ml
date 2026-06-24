from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
iris=load_iris()
x=iris.data
y=iris.target
new=[[0.5,0.4,0.6,0.3]]
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x,y)
ans=knn.predict(new)
print(f"the type of flower is :{iris.target_names[ans[0]]}")
