from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from pandas.plotting import lag_plot
#from pandas import datetime
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

# Read the CSV file
df = pd.read_csv("C:/Users/bogga/Downloads/TSLA.csv")

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Get the list of column names
columns = df.columns.tolist()

# Routes
@app.route('/')
def index():
    return render_template('index.html', columns=columns)

@app.route('/result', methods=['POST'])


def result():
    if request.method == 'POST':
        # Get the selected column from the form
        column_name = request.form['column_input']

        # Extract the selected column data
        selected_data = df[column_name].values

        # Check if the selected column contains datetime values
        if pd.api.types.is_datetime64_any_dtype(selected_data):
           # Convert datetime values to numeric values (seconds since the epoch)
           selected_data = selected_data.astype(np.int64) // 10**9

        # Split the data into training and testing sets
        train_data, test_data = selected_data[:int(len(selected_data)*0.8)], selected_data[int(len(selected_data)*0.8):]
        #train_data, test_data = df[0:int(len(df)*0.8)], df[int(len(df)*0.8):]
        # Fit ARIMA model
        order = (5, 1, 0)  # Adjust order as needed
        model = ARIMA(train_data, order=order)
        model_fit = model.fit()

        

        # Make predictions
        predictions = model_fit.forecast(steps=len(test_data)) 
        print(predictions) 
        plt.figure(figsize=(12, 7))
        plt.plot(selected_data[:len(train_data)], 'green', color='blue', label='Training Data')
        plt.plot(np.arange(len(train_data), len(train_data) + len(test_data)), predictions, color='green', marker='o', linestyle='dashed',
             label='Predicted Price')
        plt.plot(np.arange(len(train_data), len(train_data) + len(test_data)), test_data, color='red', label='Actual Price')
        plt.title('Tesla Prices Prediction')
        plt.xlabel('Dates')
        plt.xticks(np.arange(0,1857, 600), df['Date'][0:1857:600])
        plt.ylabel('Prices')
        plt.legend()

        # Save the plot to a BytesIO object
        # buffer = BytesIO()
        # plt.savefig(buffer, format='png')
        # buffer.seek(0)
        # plt.close()
        # plot_data = base64.b64encode(buffer.read()).decode('utf-8')
		# Calculate errors
        def smape_kun(y_true, y_pred):
            return np.mean((np.abs(y_pred - y_true) * 200/ (np.abs(y_pred) + np.abs(y_true))))

        error_mse = mean_squared_error(test_data, predictions)
        error_smape = smape_kun(test_data, predictions)
        print("mean square error", error_mse)
		
	

        # # Save the plot to a BytesIO object
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close()

        # Convert the plot to base64 for embedding in HTML
        plot_data = base64.b64encode(buffer.read()).decode('utf-8')


        return render_template('result.html', 
                               column_name=column_name,
                               error_mse=error_mse, 
                               error_smape=error_smape,plot_data=plot_data)
    else:
        return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)
