from flask import Flask, redirect,request,jsonify,render_template, url_for
import pickle as pl
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
application= Flask(__name__)
app=application
ridge_model=pl.load(open('model/ridge.pkl','rb'))
scaler=pl.load(open('model/scaler.pkl','rb'))

@app.route('/')
def hello_world():
    return render_template('index.html')
@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        Temp=float(request.form.get('temperature'))
        RH=float(request.form.get('RH'))
        Ws=float(request.form.get('Ws'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))
        new_data_scaled=scaler.transform([[Temp,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_data_scaled)
        return render_template('home.html',results=f'The predicted value is {result[0]:.2f}')   

    else:
        return  render_template('home.html')
        
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
   