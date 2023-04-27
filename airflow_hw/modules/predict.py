# <YOUR_IMPORTS>
import dill
import os
import json
import pandas as pd
from datetime import datetime

path = os.environ.get('PROJECT_PATH', '.')
with open(f'{path}/data/models/cars_pipe.pkl', 'rb') as file:
   model = dill.load(file)
def predict():
    # <YOUR_CODE>
    predictions = []
    for file in os.listdir(f'{path}/data/test'):
        file_path = os.path.join(f'{path}/data/test', file)
        if os.path.isfile(file_path):
            with open(file_path) as f:
                data = json.load(f)
            X = pd.DataFrame(data,index=['id'])
            y = model.predict(X)
            predictions.append({'car_id':X['id'][0],'pred':y[0]})

    predictions = pd.DataFrame(predictions)
    predictions.to_csv(os.path.join(f'{path}/data/predictions', f'preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv'),index=False)



if __name__ == '__main__':
    predict()