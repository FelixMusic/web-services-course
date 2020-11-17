from flask import Flask, request, jsonify, abort, redirect, url_for, render_template, send_file
from joblib import load
import numpy as np
import pandas as pd

app = Flask(__name__)

knn = load('knn.joblib')

@app.route('/')
def hello_world():
    print('Hi!!!')
    return '<h1>Hello, my very best friend!</h1>'

@app.route('/user/<username>')
def show_user_profile(username):
    username = float(username) * float(username)
    return 'User %s' % username

def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)

@app.route('/avg/<nums>')
def avg(nums):
    nums = nums.split(',')
    nums = [float(num) for num in nums]
    nums_mean = mean(nums)
    print(nums_mean)
    return str(nums_mean)

@app.route('/iris/<param>')
def iris(param):

    param = param.split(',')
    param = [float(num) for num in param]

    param = np.array(param).reshape(1, -1)
    predict = knn.predict(param)

    return str(predict)

@app.route('/show_image')
def show_image():

    return '<img src="/static/setosa.jpg" alt="setosa">'


@app.route('/badrequest400')
def bad_request():
    return abort(400)


@app.route('/iris_post', methods=['POST'])
def add_message():

    try:

        content = request.get_json()

        param = content['flower'].split(',')
        param = [float(num) for num in param]

        param = np.array(param).reshape(1, -1)
        predict = knn.predict(param)

        predict = {'class': str(predict[0])}
    except:
        return redirect(url_for('bad_request'))

    return jsonify(predict)





from flask_wtf import FlaskForm
from wtforms import StringField, FileField, SelectField, TextField
from wtforms.validators import DataRequired, Required
import pickle
# import xgboost as xgb
# from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

app.config.update(dict(
    SECRET_KEY="powerful secretkey",
    WTF_CSRF_SECRET_KEY="a csrf secret key"
))

class MyForm(FlaskForm):
    
    auto = SelectField('Модель:', choices=['Kia Rio I',
       'Kia Rio I Рестайлинг', 'Kia Rio II', 'Kia Rio II Рестайлинг',
       'Kia Rio III', 'Kia Rio III Рестайлинг', 'Kia Rio IV',
       'Kia Rio IV X-Line', 'Kia Rio IV Рестайлинг'])

    year = SelectField('Год выпуска:', choices=[2000, 2001, 2002, 2003, 2004, 2005,2006, 2007, 
                 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020])
    km_age = StringField('Пробег:', validators=[DataRequired()])
    engine_power = SelectField('Мощность двигателя (л/с):', choices=[75, 82, 84, 95, 96, 97, 98, 100, 105, 107, 108, 109, 112, 123])

    body_type = SelectField('Тип кузова:', choices=['седан', 'универсал 5 дв.', 'хэтчбек 5 дв.'])
    trans = SelectField('Тип трансмисии:', choices=['автоматическая', 'механическая'])
    owners_count = SelectField('Количество владельцев:', choices=['1 владелец', '2 владельца', '3 или более'])
    # file = FileField()

from werkzeug.utils import secure_filename
import os

@app.route('/submit', methods=('GET', 'POST'))
def submit():
    form = MyForm()
    if form.validate_on_submit():
        # создадим базовый фрейм с параметрами автомобиля
        my_df = pd.DataFrame({'year': [0],
                      'km_age': [0],
                      'engine_power': [0],       
                      'Kia Rio I': [0],
                      'Kia Rio I Рестайлинг': [0],
                      'Kia Rio II': [0],
                      'Kia Rio II Рестайлинг': [0], 
                      'Kia Rio III': [0], 
                      'Kia Rio III Рестайлинг': [0],
                      'Kia Rio IV': [0],
                      'Kia Rio IV X-Line': [0],
                      'Kia Rio IV Рестайлинг': [0],
                      'седан': [0],
                      'универсал 5 дв.': [0],
                      'хэтчбек 5 дв.': [0],
                      'автоматическая': [0],
                      'механическая': [0],
                      '1 владелец': [0],
                      '2 владельца': [0],
                      '3 или более': [0]
                        })

        my_df.loc[0, 'year'] = int(form.year.data)
        my_df.loc[0, 'km_age'] = int(form.km_age.data)
        my_df.loc[0, 'engine_power'] = int(form.engine_power.data)
        my_df.loc[0, form.auto.data] = 1
        my_df.loc[0, form.body_type.data] = 1
        my_df.loc[0, form.trans.data] = 1
        my_df.loc[0, form.owners_count.data] = 1


        with open('/root/random_forest.pickle','rb') as modelFile:
            model = pickle.load(modelFile)

        prediction = int(model.predict(my_df)[0])
        return 'Цена автомобиля: ' + str(prediction) + ' рублей'

    return render_template('submit.html', form=form)
