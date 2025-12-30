import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64

#Function to create a download link for the predicted CSV file
def get_binary_file_downloader_html(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="predicted_heart_disease.csv">Download CSV File</a>'
    return href

st.title("Heart Disease Predictor")
tab1, tab2, tab3 = st.tabs(["Predict", "Bulk Predict", "Model Information"])


with tab1:
    age = st.number_input("Age (years)", min_value=1, max_value=150)
    sex = st.selectbox("Sex", ["Male", "Female"])
    chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300)
    cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=0, max_value=1000)
    fasting_bs = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"])
    rest_ecg = st.selectbox("Resting ECG", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    max_hr = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=202)
    exercise_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.number_input("Oldpeak (ST depression induced by exercise)", min_value=0.0, max_value=10.0, step=0.1)
    st_slope = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])


    #conver categorical input to numerical
    sex = 0 if sex == "Male" else 1
    chest_pain_dict = ["Atypical Angina", "Non-Anginal Pain", "Asymptomatic", "Typical Angina"].index(chest_pain)
    fasting_bs = 1 if fasting_bs == "> 120 mg/dl" else 0
    rest_ecg_dict = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(rest_ecg)
    exercise_angina = 1 if exercise_angina == "Yes" else 0
    st_slope_dict = ["Upsloping", "Flat", "Downsloping"].index(st_slope)

    #create a DataFrame with user inputs
    input_data = pd.DataFrame({
        "Age": [age],
        "Sex": [sex],
        "ChestPainType": [chest_pain_dict],
        "RestingBP": [resting_bp],
        "Cholesterol": [cholesterol],
        "FastingBS": [fasting_bs],
        "RestingECG": [rest_ecg_dict],
        "MaxHR": [max_hr],
        "ExerciseAngina": [exercise_angina],
        "Oldpeak": [oldpeak],
        "ST_Slope": [st_slope_dict]
    })


    algonames = ["Decision Trees", "Logistic Regression", "Random Forest", "Support Vector Machine"]
    modelnames = ["tree.pkl", "LogisticR.pkl", "RandomForest.pkl", "SVM.pkl"]

    predictions = []  
    def predict_heart_disease(data):
        
        for modelname in modelnames:
            model = pickle.load(open(modelname, 'rb'))
            prediction = model.predict(data)
            predictions.append(prediction)
        return predictions



    #create a submit button to make prediction
    if st.button("Submit"):
        st.subheader("Prediction Results:")
        st.markdown("-----------------------")

        result = predict_heart_disease(input_data)
        for i in range(len(predictions)):
            st.subheader(algonames[i])
            if result[i][0] == 0:
                st.write("No heart disease detected.")
            else:
                st.write("Heart disease detected.")
            st.markdown("-----------------------")

with tab2:
    st.title("Upload CSV file")

    st.subheader("Instructions to note before uploading the file: ")
    st.info("""
            1.No NaN values are allowed in the file.
            2.Total 11 features in this order ("Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS", "RestingECG", "MaxHR", 
            "ExerciseAngina", "Oldpeak", "ST_Slope"). \n
            3. Check the spellings of the feature names.
            4. Feature values conventiones: \n
                -Age: age of the patient [years] \n
                -Sex: sex of the patient [0:Male, 1:Female] \n
                -ChestPainType: chest pain type [3:Typical Angina, 0:Atypical Angina, 1:Non-Anginal Pain, 2:Asymptomatic] \n
                -RestingBP: resting blood pressure [mm Hg] \n
                -Cholesterol: serum cholesterol [mg/dl] \n
                -FastingBS: fasting blood sugar [i: if FastingBS > 120 mg/dl, 0: otherwise] \n
                -RestingECG: resting electrocardiographic results [0:Normal, 1:ST-T Wave Abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), 2:Left Ventricular Hypertrophy (by Estes' criteria)] \n
                -MaxHR: maximum heart rate achieved [Numeric value between 60 and 202] \n
                -ExerciseAngina: exercise induced angina [1:Yes, 0:No] \n
                -Oldpeak: oldpeak = ST [Numeric value measured in depression] \n
                -ST_Slope: slope of the peak exercise ST segment [0:Upsloping, 1:Flat, 2:Downsloping] \n
            """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        
        input_data = pd.read_csv(uploaded_file)
        

        model = pickle.load(open("LogisticR.pkl", 'rb'))

        expected_columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 
        'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']

        if set(expected_columns).issubset(input_data.columns):
            input_data['Prediction LR'] = ''

            for i in range(len(input_data)):
                arr = input_data.iloc[i][expected_columns].values.reshape(1, -1)
                input_data.loc[i, 'Prediction LR'] = model.predict(arr)[0]

            input_data.to_csv('PredictedHeartLR.csv')

            #Display the predictions
            st.subheader("Predictions:")
            st.write(input_data)

            #Provide download link for the predicted CSV file
            st.markdown(get_binary_file_downloader_html(input_data), unsafe_allow_html=True)

        else:
            st.warning("Please make sure the uploaded csv file has the correct columns.")
    else:
        st.info("Upload a CSV file to get predictions.")

with tab3:
    import plotly.express as px 
    data = {'Decision Trees': 80.97, 'Logistic Regression' : 85.86, 'Random Forest': 86.95, 'Support Vector Machine': 84.22 }
    models = list(data.keys())
    Accuracy = list(data.values())

    df= pd.DataFrame(list(zip(models, Accuracy)), columns=['Models', 'Accuracy'])
    fig = px.bar(df, x='Models', y='Accuracy', title='Model Accuracy Comparison')
    st.plotly_chart(fig)



