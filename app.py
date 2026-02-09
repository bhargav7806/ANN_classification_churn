import streamlit as slt 
import numpy as np 
import tensorflow as tlf 
from sklearn.preprocessing import StandardScaler , LabelEncoder , OneHotEncoder
import pandas as pd 
import pickle 

model = tlf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl' , 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('label_encoder_geo.pkl' , 'rb') as file:
    oht_encoder_geo = pickle.load(file)

with open('scaler.pkl' , 'rb') as file:
    scaler = pickle.load(file)


slt.title('Customer Churn Prediction')

geography = slt.selectbox('Geography' , oht_encoder_geo.categories_[0])
gender = slt.selectbox('Gender' , label_encoder_gender.classes_)
age = slt.slider('Age' , 18 , 92)
balance = slt.number_input('Balance')
credit_score = slt.number_input('Credit Score')
estimated_salary = slt.slider('Estimated Salary')
tenure = slt.slider('Tenure')
num_of_products = slt.slider('Number of Products' , 1 , 4)
has_cr_card = slt.selectbox('Has credit card' , [0 , 1])
is_active_member = slt.selectbox('Is Active Member' , [0,1])

input_data = pd.DataFrame({
    'CreditScore' : [credit_score],          # ✅
    'Gender' : [label_encoder_gender.transform([gender])[0]],
    'Age' : [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [num_of_products],
    'HasCrCard' : [has_cr_card],
    'IsActiveMember' : [is_active_member],
    'EstimatedSalary' : [estimated_salary]  # ✅
})
geo_encoded = oht_encoder_geo.transform([[geography]])
geo_encoded_df = pd.DataFrame(geo_encoded , columns = oht_encoder_geo.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop = True) , geo_encoded_df] , axis = 1)

input_data_scaled = scaler.transform(input_data)


prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]

if prediction_prob > 0.5:
    slt.write("the churn probability is:",prediction_prob)
    slt.write('the customer is likely to churn')

else:
    slt.write("the churn probability is:",prediction_prob)
    slt.write('the customer is not liekly to churn')