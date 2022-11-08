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
    texts = [x for x in request.form.values()]
    df = pd.read_csv('snomed.csv')
    corpa = df.concept_name.to_list()
    tokens = texts[0].lower().split(' ')  
    model = pickle.load(open('pickled.pkl','rb'))

    doc_scores = model.get_scores(tokens)
    indices = np.argsort(doc_scores)[-10:]

    result_12 = f'{corpa[indices[-1]]}'
    result_12 = f'{corpa[indices[-2]]}'
    result_13 = f'{corpa[indices[-3]]}'
    result_14 = f'{corpa[indices[-4]]}'
    result_15 = f'{corpa[indices[-5]]}'
    result_16 = f'{corpa[indices[-6]]}'
    result_17 = f'{corpa[indices[-7]]}'
    result_18 = f'{corpa[indices[-8]]}'
    result_19 = f'{corpa[indices[-9]]}'
    result_20 = f'{corpa[indices[-10]]}'


    return render_template('home.html',result_12=result_12 , result_23 = result_13 , result_13 = result_13,result_14 = result_14,\
         result_15 = result_15, result_16 = result_16,result_17 = result_17,result_18 = result_18,result_19 = result_19, result_20 = result_20 )

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)