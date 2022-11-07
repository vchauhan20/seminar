from flask import Flask,request,  render_template
import pickle
import pandas as pd
import numpy as np
app = Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def calculate():
    texts = request.form.values()[0]
    df = pd.read_csv('snomed.csv')
    corpa = df.concept_name.to_list()
    tokens = texts.lower.split(' ')  
    model = pickle.load(open('pickle.pkl','rb'))

    doc_scores = model.get_scores(tokens)
    indices = np.argsort(doc_scores)[-3:]

   
    result_12 = f'{corpa[-1]}'
    result_23 = f'{corpa[-2]}'
    result_13 = f'{corpa[-3]}'
    return render_template('home.html',result_12=result_12 , result_23 = result_23 , result_13 = result_13)