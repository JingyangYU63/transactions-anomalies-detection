import numpy as np
import pandas as pd
import torch
import os
import logging
import io
import json

logger = logging.getLogger(__name__)

class SOM(torch.nn.Module):
    def __init__(self, input_dim=7, map_dim=(10, 10), num_epochs=20, learning_rate=0.1):
        super(SOM, self).__init__()
        self.input_dim = input_dim
        self.map_dim = map_dim
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weights = torch.nn.Parameter(torch.randn(map_dim[0], map_dim[1], input_dim), requires_grad=False)
        self.threshold = 0

    def forward(self, x):
        x = x.unsqueeze(0)
        dist = torch.cdist(x, self.weights).squeeze(0)
        min_dist, idx = torch.min(dist.view(-1), 0)
        map_idx = np.array(np.unravel_index(idx.item(), self.map_dim))
        return min_dist, map_idx

    def train_(self, x):
        for epoch in range(self.num_epochs):
            print("Training Progress: " + str(round(100 * epoch / self.num_epochs, 2)) + "%")
            for i in range(x.shape[0]):
                xi = x[i, :]
                min_dist, map_idx = self(xi)
                self.weights[map_idx[0], map_idx[1], :] += self.learning_rate * (xi - self.weights[map_idx[0], map_idx[1], :])
        
        min_dists = torch.zeros(x.shape[0], dtype=torch.float32)
        for i in range(x.shape[0]):
            xi = x[i, :]
            min_dist, map_idx = self(xi)
            min_dists[i] = min_dist
        self.threshold = torch.quantile(input=min_dists, q=0.95)

    def get_quantization_error(self, x):
        cluster_labels = torch.zeros(x.shape[0], dtype=torch.int32)
        min_dists = torch.zeros(x.shape[0], dtype=torch.float32)
        for i in range(x.shape[0]):
            xi = x[i, :]
            min_dist, map_idx = self(xi)
            cluster_labels[i] = map_idx[0] * self.map_dim[1] + map_idx[1]
            min_dists[i] = min_dist
        return min_dists, cluster_labels


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