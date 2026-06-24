from sklearn.neighbors import KNeighborsClassifier
x=[[1,2],[2,3],[4,5],[6,7],[7,8],[8,9]]
y=[0,0,0,1,1,1]
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x,y)
new=[[5,6]]
ans=knn.predict(new)
print(f"output after applying knn classification algorithm: {ans}")

