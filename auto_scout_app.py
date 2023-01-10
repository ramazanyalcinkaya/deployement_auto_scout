import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder


html_temp = """
	<div style ="background-color:#3d2fd6; padding:13px">
	<h1 style ="color:#f0f0f5; text-align:center; ">Auto Scout Project </h1>
	</div>
	"""
# this line allows us to display the front end aspects we have defined in the above code
st.markdown(html_temp, unsafe_allow_html = True)

# İmages of car
image = Image.open("araba.jpg")
st.image(image, use_column_width=True)


# Display Auto scout Dataset
st.header("_Auto Scout_")
df = pd.read_csv('final_scout_not_dummy2.csv')
st.write(df.head())

# Loading the models to make predictions
rf_model = pickle.load(open('rf_model_new', 'rb'))
lm= pickle.load(open('LM_model_new', 'rb'))
transformer_1 = pickle.load(open('transformer', 'rb'))


# User input variables that will be used on predictions
st.sidebar.title('Car Price Prediction')
html_temp = """
<div style="background-color:blue;padding:10px">
<h2 style="color:white;text-align:center;">Streamlit App </h2>
</div>"""
st.markdown(html_temp, unsafe_allow_html=True)

model_selected = st.sidebar.selectbox('Select the model', ('rf_model', 'lm'))
age=st.sidebar.selectbox("What is the age of your car:",(0,1,2,3))
hp=st.sidebar.slider("What is the hp_kw of your car?", 40, 300, step=5)
km=st.sidebar.slider("What is the km of your car", 0,350000, step=1000)
gearing_type=st.sidebar.radio('Select gear type',('Automatic','Manual','Semi-automatic'))
car_model=st.sidebar.selectbox("Select model of your car", ('Audi A1', 'Audi A3', 'Opel Astra', 'Opel Corsa', 'Opel Insignia', 'Renault Clio', 'Renault Duster', 'Renault Espace'))

rf_model = pickle.load(open('rf_model_new', 'rb'))
lm= pickle.load(open('LM_model_new', 'rb'))
transformer_1 = pickle.load(open('transformer', 'rb'))

my_dict = {
    "age": age,
    "hp_kW": hp,
    "km": km,
    'Gearing_Type':gearing_type,
    "make_model": car_model
    
}

df = pd.DataFrame.from_dict([my_dict])


st.header("The configuration of your car is below")
st.table(df)



# defining the function which will make the prediction using the data

def prediction(model, input_data):

	prediction = model.predict(input_data)
	return prediction

df2 = transformer_1.transform(df)

st.subheader("Press predict if configuration is okay")

if st.button("Predict"):
    if model_selected == "rf_model":
        result = prediction = rf_model.predict(df2)
        st.success("The estimated price of your car is €{}. ".format(prediction[0]))
    else :
        result = prediction = lm.predict(df2)
        st.success("The estimated price of your car is €{}. ".format(prediction[0]))
    
    
    
    
 

    


