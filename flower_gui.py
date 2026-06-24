import tkinter as tk
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)
iris=load_iris()
knn.fit(iris.data,iris.target)
window=tk.Tk()
window.title("FLOWER CLASSIFICATION")
tk.Label(window,text="sepal length").pack()
sl=tk.Entry(window)
sl.pack()

tk.Label(window,text="sepal width").pack()
sw=tk.Entry(window)
sw.pack()

tk.Label(window,text="petal length").pack()
pl=tk.Entry(window)
pl.pack()

tk.Label(window,text="petal width").pack()
pw=tk.Entry(window)
pw.pack()
def predict():
   predict=knn.predict([[float(sl.get()),float(sw.get()),float(pl.get()),float(pw.get())]])
   flower_name=iris.target_names[predict[0]]
   result.config(text=flower_name)
tk.Button(window,text="predict",command=predict).pack()
result=tk.Label(window,text="")
result.pack()
window.mainloop()

