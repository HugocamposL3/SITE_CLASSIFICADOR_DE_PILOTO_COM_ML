import pickle
from sklearn.preprocessing import LabelEncoder 
import pandas as pd
from flask import Flask, render_template, request, jsonify
import numpy as np

# Ajuste das pastas de template e assets
app = Flask(__name__, template_folder='template', static_folder='template/assets')

# Import do modelo já treinado e salvo (essa parte foi feita no jupyter notebook)
#modelo_pipeline = pickle.load(open('./models/models.pkl', 'rb'))
modelo_pipeline = pickle.load(open('Modelo F1 Python\models4.pkl', 'rb'))
#le = pickle.load(open('Modelo F1 Python\le1.pkl', 'rb'))

# Pagina principal
@app.route('/')
def home():
    return render_template("homepage.html")

# Pagina Forms que é preenchido pelo usuario
@app.route('/classification')
def classification():
    return render_template("form.html")


def get_data():
    year = request.form.get('year')
    driver_name = request.form.get('driver_name')
    #driver_name_transformado = le.transform(driver_name)
    constructor_name = request.form.get('constructor_name')
    #constructor_name_transformado = le.transform(constructor_name)


    d_dict = {'year': [year], 'driver_name': [driver_name],'constructor_name': [constructor_name]}

    return pd.DataFrame.from_dict(d_dict, orient='columns')

## Renderiza o resultado predito pelo modelo ML na Webpage
@app.route('/send', methods=['POST'])
def show_data():

    try:
        df = get_data()
        df = df[['year','driver_name','constructor_name']]

        # Faz a predição com os dados digitados pelo usuario
        prediction = modelo_pipeline.predict(df)    
        print(prediction)

        if prediction[0] == 1.0:
            outcome = 'OPAAAA ele ia ser vencendor em!'
            imagem = 'drugo.jpg'
        elif prediction[0] == 2.0:
            outcome = 'Vice Campeão nada mal em'
            imagem = 'second.jpg'
        elif prediction[0] == 3.0:
            outcome = 'Terceiro Lugar também é bom vai'
            imagem = 'terceiro.jpg'  
        else:
            outcome = 'Abaixo de terceiro lugar já não é bom em'
            imagem = 'demais.jpg'

    except ValueError as e:
        outcome = 'OPAAAA você digitou coisa errada! '+str(e).split('\n')[-1].strip()
        imagem = 'daniel.jpg'
    
    return render_template('result.html', tables=[df.to_html(classes='data', header=True, col_space=10)],
                           result=outcome, imagem=imagem)


# retorna o a predição formatada em JSON para uma solicitação HTTP
@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)

    try:
         prediction = modelo_pipeline.predict([np.array(list(data.values()))])
         output = {
        'status': 200,
        'prediction': prediction[0]
        }
         
    except ValueError as e:
        output = {
        'status': 500,
        'prediction': str(e).split('\n')[-1].strip()
        }
   
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)
