# Import flask and datetime module for showing date and time
from flask import Flask, request
import datetime
from flask_cors import CORS
import numpy as np
from collections import Counter
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pandas as pd
import io
 
# Initializing flask app
app = Flask(__name__)
CORS(app)
 
# function to perform cross validation with nearest neighbors
def cross_validation(features, data, label):
  prediction = []
  dataModified = data[:, features]

  for testIndex, testData in enumerate(dataModified):
    nearestDist = float('inf')
    nearestInd = -1
    for index, data in enumerate(dataModified):
      if testIndex == index:
        continue
      distance = np.linalg.norm(testData - data)
      if distance < nearestDist:
        nearestDist = distance
        nearestInd = index

    prediction.append(label[nearestInd])

  accuracy = np.sum(prediction == label) / len(label)

  return accuracy

def forward_selection(data, label, col_names):
  selected_features = []
  accuracy_list = []
  temp_accuracy = []

  while len(selected_features) != len(data[0]):
    for i in range(len(data[0])):
      if i in selected_features:
        temp_accuracy.append(-1)
        continue
      print('Considering feature: ', i, '(', col_names[i], ')')
      accuracy = cross_validation(selected_features + [i], data, label)
      temp_accuracy.append(accuracy)

    best_accuracy_ind = temp_accuracy.index(max(temp_accuracy))
    best_accuracy_value = max(temp_accuracy)
    selected_features.append(best_accuracy_ind)
    accuracy_list.append(best_accuracy_value)
    temp_accuracy.clear()

    print(selected_features)
    print(accuracy_list[-1])

  return [selected_features, accuracy_list]

# Route for seeing a data
@app.route('/data', methods=['GET', 'OPTIONS'])
def get_time():
 
    x = datetime.datetime.now()

    # Returning an api for showing in  reactjs
    return {
        'Name':"geek", 
        "Age":"22",
        "Date":x, 
        "programming":"python"
    }
 
@app.route('/get-features', methods=['POST', 'OPTIONS'])
def get_features():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']
    if file.filename == '':
        return 'No selected file'

    file_contents = file.stream.read()
    file.stream.seek(0)  # Reset the file cursor
    df = pd.read_csv(io.StringIO(file_contents.decode('utf-8')))

    # gets column index of classification
    col_index = 9

    class_col = df.iloc[:, col_index].values

    df.drop(df.columns[col_index], axis=1, inplace=True)

    # Preprocessing

    label_encoder = LabelEncoder()

    for column in df.select_dtypes(exclude=['number']).columns:
        df[column] = label_encoder.fit_transform(df[column])

    feature_names_list = df.columns.tolist()

    # Normalization

    min_max_scaler = MinMaxScaler()
    normalized_data = min_max_scaler.fit_transform(df)

    forward_selection(normalized_data, class_col, feature_names_list)

    return 'File uploaded successfully'

# Running app
if __name__ == '__main__':
    app.run()
