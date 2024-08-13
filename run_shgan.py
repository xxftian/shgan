import numpy as np
import pandas as pd
import random
import yaml
import json

from models.shgan.model_main import load_data_v2, run_model
from models.shgan.model import Model


if __name__ == '__main__':
    yaml_path = './config/model_cfg.yaml'
    with open(yaml_path) as file:
        args = yaml.safe_load(file)

    args['dataset'] = 'tfsd'
    args['batch_size'] = 256
    args['lr'] = 0.001
    args['embed_dim'] = 32
    args['hidden_dim'] = 128
    args['max_epochs'] = 20
    args['n_layers'] = 3

    feat_data_num, feat_data_cat, feat_data_label, g, train_idx, valid_idx = load_data_v2(
        prefix='./data/', test_size=args['test_size'])

    args['obj_tuples'] = [(col, value.max()) for col, value in feat_data_cat.items()]
    args['num_feats'] = feat_data_num.shape[1]

    run_model(args, feat_data_num, feat_data_cat, feat_data_label, g, train_idx, valid_idx)