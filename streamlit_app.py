import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import requests
from streamlit_lottie import st_lottie

# Load Lottie Animation
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

penguin_animation = load_lottie_url("https://assets9.lottiefiles.com/packages/lf20_jcikwtux.json")

# Set up page configuration
st.set_page_config(page_title="Penguin Predictor", page_icon=":penguin:", layout="wide")

# CSS for animations and custom styling
st.markdown("""
    <style>
    .fade-in-text {
        animation: fadeIn ease 3s;
    }
    @keyframes fadeIn {
        0% {opacity: 0;}
        100% {opacity: 1;}
    }

    .stApp {
        background-image: url("https://www.link-to-your-background-image.com");
        background-size: cover;
    }

    div.stButton > button:first-child {
        background-color: #00CED1;
        color: white;
        border-radius: 5px;
        height: 40px;
        width: 200px;
        font-size: 16px;
    }

    div.stButton > button:hover {
        background-color: #007c89;
        color: white;
    }

    .success-message {
        animation: slide-in 2s ease-out;
    }

    @keyframes slide-in {
        0% {
            transform: translateY(-100%);
            opacity: 0;
        }
        100% {
            transform: translateY(0);
            opacity: 1;
        }
    }

    .tooltip {
        position: relative;
        display: inline-block;
    }

    .tooltip .tooltiptext {
        visibility: hidden;
        width: 120px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px 0;
        position: absolute;
        z-index: 1;
        bottom: 125%; 
        left: 50%;
        margin-left: -60px;
        opacity: 0;
        transition: opacity 0.3s;
    }

    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    </style>
    """, unsafe_allow_html=True)

# Animated header with fade-in effect
st.markdown('<h2 class="fade-in-text" style="color: #00CED1;">💻 Discovering Penguin Species with Machine Learning</h2>', unsafe_allow_html=True)

# Lottie animation for penguins
st_lottie(penguin_animation, speed=1, height=400, key="penguin")

# Data section
with st.expander("🗂️ Data"):
    st.markdown('<h3 style="color: #00CED1;">Data</h3>', unsafe_allow_html=True)
    df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
    st.markdown('<p style="color: #00CED1;">📄 <b>Raw data</b></p>', unsafe_allow_html=True)
    st.dataframe(df)

    st.markdown('<p style="color: #00CED1;">🔢 <b>X (Features)</b></p>', unsafe_allow_html=True)
    X_raw = df.drop('species', axis=1)
    st.dataframe(X_raw)

    st.markdown('<p style="color: #00CED1;">🎯 <b>y (Target)</b></p>', unsafe_allow_html=True)
    y_raw = df.species
    st.dataframe(y_raw)

# Data visualization section
with st.expander("📊 Data visualization"):
    st.markdown('<h3 style="color: #00CED1;">Data visualization</h3>', unsafe_allow_html=True)
    st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')

# Input features in sidebar with tooltips
with st.sidebar:
    st.markdown('<h3 style="color: #00CED1;">🛠️ Input features</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="tooltip">🏝️ Island
        <span class="tooltiptext">Choose the island of the penguin</span>
    </div>
    """, unsafe_allow_html=True)
    island = st.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))

    st.markdown("""
    <div class="tooltip">📏 Bill length (mm)
        <span class="tooltiptext">Select the bill length of the penguin</span>
    </div>
    """, unsafe_allow_html=True)
    bill_length_mm = st.slider('Bill length (mm)', 32.1, 59.6, 43.9)

    bill_depth_mm = st.slider('📏 Bill depth (mm)', 13.1, 21.5, 17.2)
    flipper_length_mm = st.slider('📏 Flipper length (mm)', 172.0, 231.0, 201.0)
    body_mass_g = st.slider('⚖️ Body mass (g)', 2700.0, 6300.0, 4207.0)
    gender = st.selectbox('🚻 Gender', ('Male', 'Female'))

    # Create a DataFrame for the input features
    data = {'island': island,
            'bill_length_mm': bill_length_mm,
            'bill_depth_mm': bill_depth_mm,
            'flipper_length_mm': flipper_length_mm,
            'body_mass_g': body_mass_g,
            'sex': gender}
    input_df = pd.DataFrame(data, index=[0])
    input_penguins = pd.concat([input_df, X_raw], axis=0)

# Input features display
with st.expander("📥 Input features"):
    st.markdown('<h3 style="color: #00CED1;">Input features</h3>', unsafe_allow_html=True)
    st.markdown('<p style="color: #00CED1;">📝 <b>Input penguin data</b></p>', unsafe_allow_html=True)
    st.dataframe(input_df)
    st.markdown('<p style="color: #00CED1;">🧮 <b>Combined penguins data</b></p>', unsafe_allow_html=True)
    st.dataframe(input_penguins)

# Data Preparation Section
encode = ['island', 'sex']
df_penguins = pd.get_dummies(input_penguins, prefix=encode)

X = df_penguins[1:]
input_row = df_penguins[:1]

# Encode y (Target)
target_mapper = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}

def target_encode(val):
    return target_mapper[val]

y = y_raw.apply(target_encode)

with st.expander("⚙️ Data preparation"):
    st.markdown('<h3 style="color: #00CED1;">Data preparation</h3>', unsafe_allow_html=True)
    st.markdown('<p style="color: #00CED1;">🔢 <b>Encoded X (input penguin)</b></p>', unsafe_allow_html=True)
    st.dataframe(input_row)
    st.markdown('<p style="color: #00CED1;">🎯 <b>Encoded y (target)</b></p>', unsafe_allow_html=True)
    st.dataframe(y)

# Model Training and Prediction Section
clf = RandomForestClassifier()
clf.fit(X, y)

prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

df_prediction_proba = pd.DataFrame(prediction_proba)
df_prediction_proba.columns = ['Adelie', 'Chinstrap', 'Gentoo']

# Prediction Display Section
st.markdown('<h2 style="color: #00CED1;">🔮 Predicted Species Probability</h2>', unsafe_allow_html=True)
st.dataframe(df_prediction_proba, hide_index=True)

penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
predicted_species = penguins_species[prediction][0]

# Custom animated success message
st.markdown(f'<h3>🎉 The predicted species is: <b style="color:#00CED1;">{predicted_species}</b></h3>', unsafe_allow_html=True)


