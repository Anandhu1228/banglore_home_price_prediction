from flask import Flask, request, jsonify,render_template
from flask_cors import CORS
import resource

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "null"}})

@app.route('/')
def index():
    return render_template('app.html')

@app.route('/get_location_names_route')
def get_location_names():
    response = jsonify({
        'locations' : resource.get_location_names()
    })
    response.headers.add("Access-Control-Allow-Origin", '*')
    return response

@app.route('/predict_home_price',methods = ['POST'])
def predict_home_price():
    total_sqft = float(request.form['total_sqft'])
    location = request.form['location']
    bhk = int(request.form['bhk'])
    bath = int(request.form['bath'])

    response = jsonify({
        'estimated_price' : resource.get_estimated_price(location,total_sqft,bhk,bath)
    })
    response.headers.add("Access-Control-Allow-Origin", '*')
    return response

if __name__ =='__main__':
    print("Starting python flask server for home price prediction..")
    app.run()
