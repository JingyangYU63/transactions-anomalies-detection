import numpy as np
import pandas as pd
import torch
import os
import logging
import io
import json
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from som-train import SOM

logger = logging.getLogger(__name__)

def model_fn(model_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Loading the model.')
    model = SOM()

    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))

    model.to(device).eval()
    logger.info('Done loading model')
    return model

def input_fn(request_body, content_type='text/csv'):
    logger.info('Deserializing the input data.')
    if content_type == 'text/csv':
        # Parse the CSV data using pandas
        input_data = request_body
        df = pd.read_csv(io.StringIO(input_data), header=None)
        logger.info(f'Input data:\n{df}')
        
        # Convert the parsed data to a PyTorch tensor
        input_tensor = torch.tensor(df.values).float()
        
        return input_tensor
    raise Exception(f'Requested unsupported ContentType in content_type {content_type}')

def output_fn(prediction_output, accept='application/json'):
    logger.info('Serializing the generated output.')
    result = []
    
    for i in range(len(prediction_output)):
        pred = {'anomaly_index': prediction_output[i]}
        logger.info(f'Adding prediction: {pred}')
        result.append(pred)
    
    if accept == 'application/json':
        return json.dumps(result), accept
    raise Exception(f'Requested unsupported ContentType in Accept:{accept}')



def predict_fn(input_data, model):
    logger.info('Generating prediction based on input parameters.')
    if torch.cuda.is_available():
        input_data = input_data.cuda()

    with torch.no_grad():
        model.eval()
        min_dists, cluster_labels = model.get_quantization_error(input_data)
        anomaly_idx = (min_dists > model.threshold).nonzero(as_tuple=True)[0].tolist()

    return anomaly_idx