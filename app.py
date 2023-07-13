from flask import Flask, render_template, request, redirect,url_for
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load(open('D:\loan_prediction\classifier_model.joblib','rb'))
status = 0

@app.route('/')
@app.route('/home')
def home_page():
    return redirect(url_for('predict_loan'))

@app.route('/predict_loan', methods=['GET','POST'])
def predict_loan():
    if request.method == 'POST':
        appIncome = request.form['appincome']
        coappIncome = request.form['coappincome']
        loan_term = request.form['loanamtterm']
        loanAmount = request.form['loanamt']
        credit_history = request.form['credit']
        gender = request.form['gender'] 
        married = request.form['married']
        education = request.form['education']
        self_employed = request.form['self']
        dependents = request.form['depend']
        property_area = request.form['area']
        features = [gender,married,dependents,education,self_employed,appIncome,coappIncome,loanAmount,loan_term,credit_history,property_area]
        status = model.predict([features])
        return render_template("index.html",status=status)
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
