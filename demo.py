# import necessary libraries
# for deployment
from flask import Flask,render_template,request

# for data manipulation
import pandas as pd
# for mathematical computations
import numpy as np
#for visualization
import matplotlib.pyplot as plt
import seaborn as sns

# instatiate the flask app
app = Flask(__name__)

# define home page
@app.route('/')
@app.route('/home')
def Home_page():
    return render_template('home.html')

# define prediction page
@app.route('/prediction', methods=['GET', 'POST'])
def Prediction_page():
    if request.method == 'POST':
        # handle file upload here
        file = request.files['file']
        # do something with the uploaded file
        return "File uploaded successfully"
    else:
        return render_template('prediction.html')

if __name__ == '__main__':
    app.run(debug=True)
