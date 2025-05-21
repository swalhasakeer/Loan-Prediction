from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

# Initialize the flask app
app = Flask(__name__)
with open('LGBMRegressor_model.pkl','rb') as f:
    reg_model = pickle.load(f)

with open('Scaler.pkl','rb') as f:
    scaler = pickle.load(f)
    
emp_map = {'Employed':0, 'Self-Employed':1, 'Unemployed':2}
edu_map = {'High School':0,'Associate':1,'Bachelor':2,'Master':3,'Doctorate':4}
features_to_log1p = ['LoanAmount','MonthlyIncome','NetWorth']
num_cols_to_standardize = [
    'Age', 'CreditScore', 'LoanAmount', 'LoanDuration',
    'CreditCardUtilizationRate', 'LengthOfCreditHistory',
    'MonthlyIncome', 'NetWorth', 'InterestRate'
]
@app.route ('/')
def home():
    return render_template('form.html', prediction_text='')

@app.route ('/Predict', methods=['POST'])
def index():
    input_data = {
    'Age' : int(request.form['Age']),
    'CreditScore' : float(request.form['CreditScore']),
    'EmploymentStatus' : request.form['EmploymentStatus'],
    'EducationLevel' : request.form['EducationLevel'],
    'LoanAmount' : int(request.form['LoanAmount']),
    'LoanDuration' : int(request.form['LoanDuration']),
    'CreditCardUtilizationRate' : float(request.form['CreditCardUtilizationRate']),
    'BankruptcyHistory' : int(request.form['BankruptcyHistory']),
    'PreviousLoanDefaults' : int(request.form['PreviousLoanDefaults']),
    'LengthOfCreditHistory' : int(request.form['LengthOfCreditHistory']),
    'MonthlyIncome' : int(request.form['MonthlyIncome']),
    'NetWorth' : float(request.form['NetWorth']),
    'InterestRate' : float(request.form['InterestRate'])
    }
    
    input_df = pd.DataFrame([input_data])
    # Encoding
    input_df['EmploymentStatus'] = input_df['EmploymentStatus'].map(emp_map)
    input_df['EducationLevel'] = input_df['EducationLevel'].map(edu_map)
    
    #log1p transform
    input_df[features_to_log1p] = input_df[features_to_log1p].apply(np.log1p)
    
    #Standardize
    input_df[num_cols_to_standardize] = scaler.transform(input_df[num_cols_to_standardize])
    
    #Predict
    risk_score = reg_model.predict(input_df)[0]
    risk_score = round(risk_score,2)
    
    return render_template('form.html',prediction_text = risk_score)
if __name__ == '__main__':
    app.run(debug=True)