from flask import Flask, render_template, request
from numpy.lib.nanfunctions import nanstd
app = Flask(__name__)
import pickle


# open a file, where you stored the pickled data
file = open('model.pkl', 'rb')
clf = pickle.load(file)
file.close()



@app.route('/')
def home():
    return render_template('dashboard.html')

@app.route('/chest', methods=["GET", "POST"])
def hello_world():
    if request.method == "POST":
        
        Fever = request.form['Fever']
        Age = request.form['Age']
        DryCough = request.form['DryCough']
        Difficulty_In_Breathing=request.form['Difficulty_In_Breathing']
        Pain = request.form['Pain']
        Nasal_Congestion = request.form['Nasal_Congestion']
        Congestion = request.form['Congestion']
        Diarrhea=request.form['Diarrhea']
        # Code for inference
        Features = [Fever,Age,DryCough,Difficulty_In_Breathing,Pain,Nasal_Congestion,Congestion,Diarrhea]
        infProb = clf.predict_proba([Features])[0][1]
        print(infProb)
        return render_template('show.html', inf=round(infProb*100))
    return render_template('index.html')
    


if __name__ == "__main__":
    app.run(debug=True)