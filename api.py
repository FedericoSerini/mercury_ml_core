import json

import pandas as pd
from flask import Flask, request
import tensorflow.keras as keras
import numpy as np

app = Flask(__name__)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@app.route('/ml/predict/<ticker>', methods=["POST"])
def ml_predict(ticker):
    content = request.get_json()
    model = keras.models.load_model('../mercury/'+ticker+'.h5')
    df = pd.DataFrame(content['imageData']).values
    a = df.reshape(1, 15, 15, 1)
    b = model.predict(a)
    c = dict({"HOLD": str(b[0][0]), "BUY": str(b[0][1]), "SELL": str(b[0][2])})
    return {'status': 'ok', 'result': c}


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=9999)
