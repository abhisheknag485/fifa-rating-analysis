import streamlit as st
import pandas as pd
import numpy as np
import pickle
import numpy as np

label_encoded_position = pickle.load(open("label_encoder_position.pickle", "rb"))

model = pickle.load(open("model.pickle", "rb"))

def predmodel():
    poss = ('FW','MF','DF','GK')
    mp = st.number_input("Enter matches played: ")
    goals = st.number_input("Enter Goal(s): ")
    assists = st.number_input("Enter Assist(s): ")
    passComp = st.number_input("Enter pass completion percentage: ")
    crossComp = st.number_input("Enter cross completion percentage: ")
    dribbleComp = st.number_input("Enter dribble completion percentage: ")
    pos = st.selectbox("Select Position",poss)
    pos_le = label_encoded_position.transform([pos])
    
    ok = st.button("Predict Total Crop Production")
    if ok:
        rating = model.predict([[mp,goals,assists,passComp,crossComp,dribbleComp,pos_le]])
        st.write("Predicted potential FIFA rating of the player: ",rating)


def show_predict_page():
    st.title("FIFA 22 player rating analysis based on their real-life performance")

    st.write("""### Developed By Abhishek Nag""")

    predmodel()


show_predict_page()

