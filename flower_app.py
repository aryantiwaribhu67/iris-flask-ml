from flask import Flask,request,render_template
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
app=Flask(__name__)
knn=KNeighborsClassifier(n_neighbors=3)
iris=load_iris()
knn.fit(iris.data,iris.target)
@app.route("/")
def home():
   return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():
   sl=float(request.form["sl"])
   sw=float(request.form["sw"])
   pl=float(request.form["pl"])
   pw=float(request.form["pw"])
   prediction=knn.predict([[sl,sw,pl,pw]])
   flower_name=iris.target_names[prediction[0]]
   return render_template("index.html",result=flower_name)
if __name__=='__main__':
 app.run(debug=True)

