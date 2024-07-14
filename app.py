import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)
model = pickle.load(open('notebook/heart_disease_nb_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    prediction = model.predict(np.array(list(data.values())).reshape(1, -1))
    output = prediction[0]
    return str(output)

@app.route('/predict_heart', methods=['GET', 'POST'])
def predict_heart():
    if request.method == 'POST':
        try:
            int_features = [int(x) for x in request.form.values()]
            final_features = [np.array(int_features)]
            prediction = model.predict(final_features)
            output = prediction[0]
            return render_template('predict_heart.html', prediction_text=f'Patient affected by Heart Diseases or Not: {output}')
        except Exception as e:
            print("Error during prediction:", e)
            return render_template('predict_heart.html', prediction_text='Error during prediction')
    else:
        return render_template('predict_heart.html')

if __name__ == "__main__":
    app.run(debug=True)
