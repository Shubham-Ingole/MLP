import sklearn
import simplexml
from flask import request,make_response, Flask
from flask_restful import Resource, Api
import numpy as np
import keras 
from keras.models import Model, load_model

app = Flask(__name__)
api = Api(app, default_mediatype='application/json')

@api.representation('application/xml')
def xml(data, code, headers=None):
    resp = make_response(simplexml.core.dumps({'response': data}), code)
    resp.headers.extend(headers or {})
    return resp


import pickle as pickle

autoencoder = load_model('autoencoder.h5')

class Predict(Resource):

    def get(self):
        mod5_attribs=[]
        #mod5_attribs.append(request.args.get('mod5_attrib0',type=int))
        mod5_attribs.append(request.args.get('mod5_attrib1',type=float))
        mod5_attribs.append(request.args.get('mod5_attrib2',type=float))
        mod5_attribs.append(request.args.get('mod5_attrib3',type=float))
        mod5_attribs.append(request.args.get('mod5_attrib4',type=float))
        mod5_attribs.append(request.args.get('mod5_attrib5',type=float))
        mod5_attribs.append(request.args.get('mod5_attrib6',type=float))
        mod5_attribs.append(request.args.get('mod5_attrib7',type=float))
        mod5_attribs.append(request.args.get('mod5_attrib8',type=float))
        mod5_attribs.append(request.args.get('mod5_attrib9',type=float))
        mod5_attribs.append(request.args.get('mod5_attrib10',type=float))
        mod5_attribs.append(request.args.get('mod5_attrib11',type=float))
        mod5_attribs.append(request.args.get('mod5_attrib12',type=float))
        mod5_attribs.append(request.args.get('mod5_attrib13',type=float))
        mod5_attribs.append(request.args.get('mod5_attrib14',type=float))
        mod5_attribs.append(request.args.get('mod5_attrib15',type=float))
        mod5_attribs.append(request.args.get('mod5_attrib16',type=float))
        mod5_attribs.append(request.args.get('mod5_attrib17',type=float))
        mod5_attribs.append(request.args.get('mod5_attrib18',type=float))
        mod5_attribs.append(request.args.get('mod5_attrib19',type=float))
        mod5_attribs.append(request.args.get('mod5_attrib20',type=float))
        mod5_attribs.append(request.args.get('mod5_attrib21',type=float))
        mod5_attribs.append(request.args.get('mod5_attrib22',type=float))
        mod5_attribs.append(request.args.get('mod5_attrib23',type=float))
        mod5_attribs.append(request.args.get('mod5_attrib24',type=float))
        mod5_attribs.append(request.args.get('mod5_attrib25',type=float))
        mod5_attribs.append(request.args.get('mod5_attrib26',type=float))
        mod5_attribs.append(request.args.get('mod5_attrib27',type=float))
        mod5_attribs.append(request.args.get('mod5_attrib28',type=float))
        mod5_attribs.append(request.args.get('mod5_attrib29',type=float))
       
  
        data = np.array(list(data.values()))
        data = np.reshape(data, newshape=(1, 29), order='C')
        yPred1 = autoencoder.predict(data)
        mse = np.mean(np.power(data - yPred1, 2), axis=1)
        threshold = 0.1
        y_pred = [1 if e > threshold else 0 for e in mse]
        output = y_pred[0]
        return { 'result ' : int(output) }







api.add_resource(Predict,'/predict')




if __name__ == '__main__':
    import os
    HOST = os.environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(os.environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
    app.run(HOST, PORT)
