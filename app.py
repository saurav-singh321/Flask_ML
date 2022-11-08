import pickle
from flask import Flask,request,app,url_for,render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
#Load the model
spaceship_model = pickle.load(open('spaceship.pkl','rb'))
car_model       = pickle.load(open('car_model.pkl','rb'))
mobile_model    = pickle.load(open('mobile.pkl','rb'))
house_model     = pickle.load(open('house_price.pkl','rb'))
titanic_model   = pickle.load(open('titanic.pkl','rb')) 

@app.route('/') 
def home():
    return render_template('base.html')


# This function is to display the form of spaceship
@app.route('/spaceship',methods=['GET'])
def spaceship():
        return render_template('spaceship.html')

# This function is to display the output of spaceship
@app.route('/predict_spaceship',methods=['POST'])
def predict_spaceship():
    data = [x for x in request.form.values()] # whatever we get data, it is stored in data
    new_data = np.array(data).reshape(1,-1)
    output = spaceship_model.predict(new_data)[0]
    output = bool(output)
    if output == True:
        return render_template('spaceship.html',prediction_text = f"The Spaceship may reached to its destination!")
    else:
        return render_template('spaceship.html',prediction_text = f"The Spaceship may not reached to its destination")


# This function is to display the form of car
@app.route('/car',methods=['GET'])
def car():
        return render_template('car.html')

# This function is to display the output of car
@app.route('/predict_car',methods=['POST'])
def predict_car():
    data = [x for x in request.form.values()] # whatever we get data, it is stored in data
    new_data = np.array(data).reshape(1,-1)
    output = car_model.predict(new_data)[0]
    return render_template('car.html',prediction_text = f"The predicted price of the car is {output} Lakhs")

# This function is to display the form of mobile
@app.route('/mobile',methods=['GET'])
def mobile():
        return render_template('mobile.html')

# This function is to display the output of mobile
@app.route('/predict_mobile',methods=['POST'])
def predict_mobile():
    data = [x for x in request.form.values()] # whatever we get data, it is stored in data
    new_data = np.array(data).reshape(1,-1)
    output = mobile_model.predict(new_data)[0]
    return render_template('mobile.html',prediction_text = f"The predicted price of the Mobile is {output} Thousand")
    

# This function is to display the form of house price
@app.route('/house',methods=['GET'])
def house():
        return render_template('house.html')

# This function is to display the output of house price
@app.route('/predict_house',methods=['POST'])
def predict_house():
    data = [x for x in request.form.values()] # whatever we get data, it is stored in data
    new_data = np.array(data).reshape(1,-1)
    output = house_model.predict(new_data)[0]
    return render_template('house.html',prediction_text = f"The predicted price of the house is {output}")
    

# This function is to display the form of titanic
@app.route('/titanic',methods=['GET'])
def titanic():
        return render_template('titanic.html')

# This function is to display the output of titanic survival
@app.route('/predict_titanic',methods=['POST'])
def predict_titanic():
    data = [x for x in request.form.values()] # whatever we get data, it is stored in data
    new_data = np.array(data).reshape(1,-1)
    output = house_model.predict(new_data)[0]
    return render_template('titanic.html',prediction_text = f"The predicted price of the house is {output}")
    
















if __name__ == '__main__':
    app.run(debug=True)
