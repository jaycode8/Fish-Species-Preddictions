# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 11:09:23 2022

@author: JAYMOH
"""

import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import pickle
import time


#======================== setting up the page configurations =================================
st.set_page_config(
        page_title=('Fish Species prediction'),
        #page_icon='house'
    )

selected = option_menu(
        menu_title = None,
        options = ['Home','Classification'],
        icons = ['house','book'],
        default_index = 0,
        orientation = 'horizontal'
    )

dataset = pd.read_csv("../Model/Fish.csv")
randomForest = pickle.load(open("../Model/FishRandomForestModel.sav",'rb'))

def homePage():
    st.markdown('<h2 style="text-align:center; color:#f39c12; text-decoration:underline; font-family:garamond;">Fish Species Prediction Web App using Random Forest Classifier</h2>', unsafe_allow_html=True)
    #st.subheader('Fish Species Prediction Web App using Random Forest Classifier')
    #================ pic or vida ===============================
    st.image('./resources/fish01.gif', width=500)
    
    
    #========== data set sample ===============================
    st.caption('Dataset Sample')
    st.write(dataset.head(5))
    st.caption('Dataset number of rows and columns')
    st.text(dataset.shape)
    st.caption('Available Species in the dataset')
    st.text(dataset['Species'].value_counts())
    
def predictions():
    st.subheader('Provide with the following required fish input specifications....')
    #========inputs===========
    Weight = st.slider('Weight',0,1650)
    Length1 = st.slider('Length1',7.5,59.0)
    Length2 = st.slider('Length2',8.4,63.4)
    Length3 = st.slider('Length3',8.8, 68.0)
    Height = st.slider('Height',1.7284, 18.957)
    Width = st.slider('Width', 1.0476, 8.142)
    
    input_data = ([Weight,Length1,Length2,Length3,Height,Width])
    
    #=========convert data into numpy array===========
    input_data = np.asarray(input_data)
    #===========input data reshape===============
    input_data = input_data.reshape(1, -1)
    
    pred = ''
    
    if st.button('Classify'):
        with st.spinner('Wait for it ...'):
            time.sleep(5)
            
        pred = randomForest.predict(input_data)
    
        if pred[0] == 0:
            st.success('Fish species is Bream')
            
        elif pred[0] == 1:
            st.success('Fish species is Parkki')
            
        elif pred[0] ==2:
            st.success('Fish species is Perch')
            
        elif pred[0] == 3:
            st.success('Fish species is Pike')
            
        elif pred[0] == 4:
            st.success('Fish species is Roach')
            
        elif pred[0] == 5:
            st.success('Fish species is Smelt')
            
        else:
            st.success('Fish species is Whitefish')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write('')
        with col2:
            st.image('./resources/fish02.gif')
        with col3:
            st.write('')


if selected == 'Home':
    homePage()

elif selected == 'Classification':
    predictions()













