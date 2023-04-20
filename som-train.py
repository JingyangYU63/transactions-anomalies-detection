import subprocess

# Install boto3 module
subprocess.check_call(['pip', 'install', 'boto3'])

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import argparse
import os
import boto3
import csv
import io

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
                dist_to_winner = torch.cdist(torch.tensor([map_idx], dtype=torch.float32), \
                                             torch.tensor([[(x, y) for x in range(self.map_dim[0])] for y in range(self.map_dim[1])], dtype=torch.float32)).squeeze()
                lr = self.learning_rate * torch.exp(-dist_to_winner)
                self.weights += lr.unsqueeze(-1) * (xi - self.weights)
                
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

    

if __name__ == '__main__':
    
    os.environ["SM_MODEL_DIR"] = "/opt/ml/model/"
    os.environ["SM_CHANNEL_TRAINING"] = "s3://sagemaker-us-east-2-216657257678/pytorch/df_normalized.csv"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    
    args, _ = parser.parse_known_args()

    num_epochs    = args.epochs
    learning_rate = args.learning_rate
    batch_size    = args.batch_size
    model_dir     = args.model_dir
    data_dir      = args.data_dir
    
    data_dir_list = data_dir.split('/')
    local_dir_path = "./dataset"
    os.makedirs(local_dir_path, exist_ok = True)
    s3 = boto3.client('s3')
    file_name = data_dir_list[-1]
    local_file_path = local_dir_path + "/" + file_name
    bucket = data_dir_list[2]
    data_key = data_dir_list[-2] + '/' + data_dir_list[-1]
    s3.download_file(bucket, data_key, local_file_path)
    
    # opening the CSV file
    with open(local_file_path, mode ='r') as file:

        # reading the CSV file
        csvFile = list(csv.reader(file))[1:]
        
    df_normalized_tolist = [[float(e) for e in row[1:]] for row in csvFile[1:]]
    X = torch.tensor(df_normalized_tolist, dtype=torch.float32, requires_grad=False)

    # Define the input and map dimensions
    input_dim = X.shape[1]
    map_dim = (10, 10)

    # Create an instance of the SOM module
    som = SOM(input_dim, map_dim, num_epochs, learning_rate)

    # Train the SOM module
    torch.manual_seed(0)
    som.train_(X)

    # Get the cluster labels for each input record
    min_dists, cluster_labels = som.get_quantization_error(X)

    threshold = torch.quantile(input=min_dists, q=0.95)
    anomaly_idx = (min_dists > threshold).nonzero(as_tuple=True)[0].tolist()
    model_path = '/opt/ml/model/model.pth'
    torch.save(som.state_dict(), model_path)