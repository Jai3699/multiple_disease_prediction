from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd


app = Flask(__name__)
decisiontree=pickle.load(open('/config/workspace/pickle/dtc_diabetes.pkl','rb'))
scaler_heart_stroke=pickle.load(open('/config/workspace/pickle/scaler_heart_stroke.pkl','rb'))
scaler_hypertension=pickle.load(open('/config/workspace/pickle/scaler_hypertension.pkl','rb'))
scaler_migraine=pickle.load(open('/config/workspace/pickle/scaler_migraine.pkl','rb'))
rfc_heart_stroke=pickle.load(open('/config/workspace/pickle/rfc_heart_stroke.pkl','rb'))
rfc_hypertension=pickle.load(open('/config/workspace/pickle/rfc_hypertension.pkl','rb'))
rfc_migraine=pickle.load(open('/config/workspace/pickle/rfc_migraine.pkl','rb'))
@app.route("/")
def index():
    return render_template('index.html')



@app.route('/enter', methods=['POST'])
def enter_symptoms():
    if request.method == 'POST':

        disease = request.form.get('disease')
        
        if disease == 'diabetes':

            return render_template('diabetes.html')
                

        elif disease=='hypertension':
            return render_template('hypertension.html')  
        elif disease=='migraine':
            return render_template('migraine.html')
        elif disease=='heart':
            return render_template('heartstroke.html')

        
        

        else:
            return render_template('index.html')             





@app.route('/diabetes',methods=['POST'])
def fun():

    if request.method=='POST':
        Pregnancies=int(request.form['Pregnancies'])
        print(Pregnancies)
        Glucose=float(request.form['Glucose'])
        print(Glucose)
        BloodPressure=float(request.form['BloodPressure'])
        print(BloodPressure)
        SkinThickness=float(request.form['SkinThickness'])
        Insulin=float(request.form['Insulin'])
        BMI=float(request.form['BMI'])
        DiabetesPedigreeFunction=float(request.form['DiabetesPedigreeFunction'])
        Age=float(request.form['Age'])
        outcome=decisiontree.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        if outcome[0]==1:
            result='Person is Diabetic'
            return render_template('single_prediction.html',result=result)
        else:
            result='Person is not Diabetic'
            return render_template('single_prediction.html',result=result)

@app.route('/heartstroke',methods=['POST'])
def fun1():

    if request.method=='POST':
        gender=int(request.form['gender'])
        age=float(request.form['age'])
        hypertension=float(request.form['hypertension'])
        heart_disease=float(request.form['heart_disease'])
        ever_married=float(request.form['ever_married'])
        work_type=float(request.form['work_type'])
        Residence_type=float(request.form['Residence_type'])
        avg_glucose_level=float(request.form['avg_glucose_level'])
        bmi=float(request.form['bmi'])
        smoking_status=float(request.form['smoking_status'])
        scaled=scaler_heart_stroke.transform([[gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status]])
        outcome=rfc_heart_stroke.predict(scaled)
        if outcome[0]==1:
            result='Person has chances of Heart Stroke'
            return render_template('single_prediction.html',result=result)
        else:
            result='Person has not chances of Heart Stroke'
            return render_template('single_prediction.html',result=result)


@app.route('/hypertension',methods=['POST'])
def fun2():
    
    if request.method=='POST':        
        education=int(request.form['education'])
        age=float(request.form['age'])
        BMI=float(request.form['BMI'])
        currentSmoker=float(request.form['currentSmoker'])
        heartRate=float(request.form['heartRate'])
        scaled=scaler_hypertension.transform([[education,age,BMI,currentSmoker,heartRate]])
        outcome=rfc_hypertension.predict(scaled)
        if outcome[0]==1:
            result='Person has Hypertension'
            return render_template('single_prediction.html',result=result)
        else:
            result='Person has not Hypertension'
            return render_template('single_prediction.html',result=result)    


@app.route('/migraine',methods=['POST'])
def fun3():

    if request.method=='POST':
        Age=int(request.form['Age'])
        Duration=float(request.form['Duration'])
        Frequency=float(request.form['Frequency'])
        Location=float(request.form['Location'])
        Intensity=float(request.form['Intensity'])
        Nausea=float(request.form['Nausea'])
        Vomit=float(request.form['Vomit'])
        Phonophobia=float(request.form['Phonophobia'])
        Photophobia=float(request.form['Photophobia'])
        Visual=float(request.form['Visual'])
        Sensory=float(request.form['Sensory'])
        Conscience=float(request.form['Conscience'])
        scaled=scaler_migraine.transform([[Age,Duration,Frequency,Location,Intensity,Nausea,Vomit,Phonophobia,Photophobia,Visual,Sensory,Conscience]])
        outcome=rfc_migraine.predict(scaled)
        for i in range(len(outcome)):
            if outcome[i]==0:
                result='Person has Basilar-type aura migraine'
                return render_template('single_prediction.html',result=result)
            elif outcome[i]==1:
                result='Person has Familial hemiplegic migraine'
                return render_template('single_prediction.html',result=result)
            elif outcome[i]==2:
                result='Person has Migraine without aura'
                return render_template('single_prediction.html',result=result)  

            elif outcome[i]==3:
                result='Person has Other problem'
                return render_template('single_prediction.html',result=result) 
            elif outcome[i]==4:
                result='Person has Sporadic hemiplegic migraine'
                return render_template('single_prediction.html',result=result)  
            elif outcome[i]==5:
                result='Person has Typical aura with migraine'
                return render_template('single_prediction.html',result=result)     
            elif outcome[i]==6:
                result='Person has Typical aura without migraine'
                return render_template('single_prediction.html',result=result)      


if __name__ == '__main__':

    app.run(host='0.0.0.0')
    app.run(debug=True)