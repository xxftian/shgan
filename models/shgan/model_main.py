import pickle
import os
import scipy.sparse as sp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import dgl
import time
import warnings

from collections import defaultdict
from scipy.io import loadmat
from tqdm import tqdm
from easydict import EasyDict
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

warnings.filterwarnings('ignore')


def load_data_v2(prefix: str, test_size: float = 0.2):
    nodes_df = pd.read_csv(os.path.join(prefix, 'features_v2.txt'), sep='\t').sort_values('device_number',
                                                                                          ascending=True).reset_index(
        drop=True)
    adj_v = pd.read_csv(os.path.join(prefix, 'adj_v2.csv'))
    adj_p = pd.read_csv(os.path.join(prefix, 'adj_p2.csv'))
    adj_r = pd.read_csv(os.path.join(prefix, 'adj_r2.csv'))

    edge_connections = {
        ('user', 'rel_v', 'user'): (torch.from_numpy(adj_v['device_number_a'].values).type(torch.long),
                                    torch.from_numpy(adj_v['device_number_b'].values).type(torch.long)),
        ('user', 'rel_p', 'user'): (torch.from_numpy(adj_p['device_number_a'].values).type(torch.long),
                                    torch.from_numpy(adj_p['device_number_b'].values).type(torch.long)),
        ('user', 'rel_r', 'user'): (torch.from_numpy(adj_r['device_number_a'].values).type(torch.long),
                                    torch.from_numpy(adj_r['device_number_b'].values).type(torch.long)),
    }
    edge_feats = {
        ('user', 'rel_v', 'user'): torch.from_numpy(adj_v.iloc[:, 2:].values).float(),
        ('user', 'rel_p', 'user'): torch.from_numpy(adj_p.iloc[:, 2:].values).float(),
        ('user', 'rel_r', 'user'): torch.from_numpy(adj_r.iloc[:, 2:].values).float()
    }
    etypes = ['rel_v', 'rel_p', 'rel_r']

    # 1.初始化异质图
    hetero_graph = dgl.heterograph(edge_connections)

    # 2.添加边特征
    for etype in edge_feats:
        hetero_graph.edges[etype].data['feat'] = edge_feats[etype]

    # 3.重新组pair对
    edges_data = []
    for etype in hetero_graph.canonical_etypes:
        src, dst = hetero_graph.edges(etype=etype)
        feats = hetero_graph.edges[etype].data['feat']
        edges_data.append((src, dst, feats, etype))  # 每种关系

    feat_size = {etype[1]: hetero_graph.edges[etype].data['feat'].shape[1] for etype in hetero_graph.canonical_etypes}
    total_feat_size = sum(feat_size.values())
    homogeneous_edges = []

    # 4.抽取edges_data
    for src, dst, feats, etype in edges_data:
        rel_v_feats = feats if etype == 'rel_v' else torch.zeros(feats.shape[0], feat_size['rel_v'])
        rel_p_feats = feats if etype == 'rel_p' else torch.zeros(feats.shape[0], feat_size['rel_p'])
        rel_r_feats = feats if etype == 'rel_r' else torch.zeros(feats.shape[0], feat_size['rel_r'])

        combined_feats = torch.cat((rel_v_feats, rel_p_feats, rel_r_feats), dim=1)
        homogeneous_edges.append((src, dst, combined_feats))  # 当前边的所有特征

    # 5.合并所有边
    src_all, dst_all, edge_all = zip(*homogeneous_edges)
    #     print(src_all[0].shape, dst_all[0].shape, edge_all[0].shape)
    #     print(src_all[1].shape, dst_all[1].shape, edge_all[1].shape)
    #     print(src_all[2].shape, dst_all[2].shape, edge_all[2].shape)

    src_all = torch.cat(src_all)
    dst_all = torch.cat(dst_all)
    edge_all = torch.cat(edge_all)

    # 6.构造同质图
    g = dgl.graph((src_all, dst_all))
    # homogeneous_graph.ndata['feat'] = torch.from_numpy(feat_df.values)
    g.edata['feat'] = edge_all

    obj_feas = ['user_product_type_5g', 'credit_class', 'brand_id', 'channel_type', 'is_acct', 'mon_is_acct']
    num_feas = ['total_fee', 'vas_sms_fee', 'vas_flux_fee', 'other_fee', 'mon_total_fee']
    feat_data_num = nodes_df[num_feas].values
    feat_data_cat = dict(zip(obj_feas, nodes_df[obj_feas].values.T))
    feat_data_label = nodes_df['label'].apply(lambda x: 2 if x < 0 else x).values

    index = list(range(len(feat_data_label)))
    train_idx, valid_idx, y_train, y_valid = train_test_split(index, feat_data_label, stratify=feat_data_label,
                                                              test_size=test_size,
                                                              random_state=42, shuffle=True)

    g.ndata['label'] = torch.from_numpy(feat_data_label).to(torch.long)

    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)

    feat_data_num = torch.from_numpy(feat_data_num).type(torch.float32)
    feat_data_label = torch.from_numpy(feat_data_label).type(torch.int32).long()
    feat_data_cat = {k: torch.from_numpy(v).type(torch.int32) for k, v in feat_data_cat.items()}

    train_idx = torch.from_numpy(np.array(train_idx)).type(torch.int32)
    valid_idx = torch.from_numpy(np.array(valid_idx)).type(torch.int32)

    return feat_data_num, feat_data_cat, feat_data_label, g, train_idx, valid_idx


def load_model_input(numeric_feat, object_feat, labels, dst_nodes, src_nodes, device, blocks):
    batch_num_inputs = numeric_feat[src_nodes].to(device)
    batch_cat_inputs = {i: object_feat[i][src_nodes].to(device) for i in object_feat.keys()}

    batch_dst_labels = labels[dst_nodes].to(device)
    batch_src_labels = labels[src_nodes].to(device)
    batch_src_labels[:batch_dst_labels.shape[0]] = 2

    return batch_num_inputs, batch_cat_inputs, batch_dst_labels, batch_src_labels


def run_model(args, feat_data_num, feat_data_cat, feat_data_label, graph, train_idx, valid_idx):
    graph = graph.to(args['device'])
    train_idx = train_idx.to(torch.long).to(args['device'])
    valid_idx = valid_idx.to(torch.long).to(args['device'])
    feat_data_num = feat_data_num.to(args['device'])
    feat_data_cat = {k: v.to(args['device']) for k, v in feat_data_cat.items()}
    feat_data_label = feat_data_label.to(args['device'])

    test_predictions = torch.from_numpy(
        np.zeros([len(feat_data_num), 2])).float().to(args['device'])

    y_target = feat_data_label[train_idx]
    y = feat_data_label
    loss_fn = nn.CrossEntropyLoss().to(args['device'])

    sinkhorn = SamplesLoss(loss='sinkhorn',
                           p=args['sh_p'],
                           scaling=args['sh_scaling'],
                           blur=args['sh_blur']).to(args['device'])
    # triloss = TripletLoss(args['device'])

    train_sampler = MultiLayerFullNeighborSampler(args['n_layers'])
    train_dataloader = NodeDataLoader(
        graph,
        train_idx,
        train_sampler,
        device=args['device'],
        batch_size=args['batch_size'],
        shuffle=True,
        drop_last=False,
        num_workers=0
    )

    valid_sampler = MultiLayerFullNeighborSampler(args['n_layers'])
    valid_dataloader = NodeDataLoader(
        graph,
        valid_idx,
        valid_sampler,
        device=args['device'],
        batch_size=args['batch_size'],
        shuffle=True,
        drop_last=False,
        num_workers=0
    )

    model = Model(args).to(args['device'])
    lr = args['lr'] * np.sqrt(args['batch_size'] / 1024)
    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args['wd'])
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=args['wd'], momentum=0.9, nesterov=True)
    lr_scheduler = MultiStepLR(optimizer=optimizer, milestones=[4000, 12000], gamma=0.3)
    earlystoper = early_stopper(patience=args['early_stopping'], verbose=True)

    for epoch in range(args['max_epochs']):
        train_loss_list = []
        model.train()
        for step, (src_nodes, dst_nodes, blocks) in enumerate(train_dataloader):
            batch_num_inputs, batch_cat_inputs, batch_dst_labels, batch_src_labels = load_model_input(feat_data_num,
                                                                                                      feat_data_cat,
                                                                                                      feat_data_label,
                                                                                                      dst_nodes,
                                                                                                      src_nodes,
                                                                                                      blocks=blocks,
                                                                                                      device=args[
                                                                                                          'device'])
            blocks = [block.to(args['device']) for block in blocks]
            train_batch_logits, train_batch_embs = model(blocks, batch_num_inputs, batch_src_labels, batch_cat_inputs)

            mask = batch_dst_labels == 2  # dst和unlabel部分
            train_batch_logits = train_batch_logits[~mask]
            batch_dst_labels = batch_dst_labels[~mask]
            train_batch_embs = train_batch_embs[~mask]

            if batch_dst_labels.shape[0] == 0:
                continue

            train_loss = loss_fn(train_batch_logits, batch_dst_labels)  # 这里的nan可能是因为去掉2以后没有值导致

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            train_loss_list.append(train_loss.cpu().detach().numpy())

            if step % 100 == 0:
                tr_batch_pred = torch.sum(
                    torch.argmax(train_batch_logits.clone().detach(), dim=1) == batch_dst_labels) / \
                                batch_dst_labels.shape[0]
                score = torch.softmax(train_batch_logits.clone().detach(), dim=1)[:, 1].cpu().numpy()

                try:
                    print('In epoch:{:03d}|batch:{:04d}, train_loss:{:4f}, '
                          'train_ap:{:.4f}, train_acc:{:.4f}, train_auc:{:.4f}'.format(epoch, step,
                                                                                       np.mean(train_loss_list),
                                                                                       average_precision_score(
                                                                                           batch_dst_labels.cpu().numpy(),
                                                                                           score),
                                                                                       tr_batch_pred.detach(),
                                                                                       roc_auc_score(
                                                                                           batch_dst_labels.cpu().numpy(),
                                                                                           score)))
                except:
                    pass

        val_loss_list = 0
        val_acc_list = 0
        val_all_list = 0
        model.eval()
        with torch.no_grad():
            for step, (src_nodes, dst_nodes, blocks) in enumerate(valid_dataloader):
                batch_num_inputs, batch_cat_inputs, batch_dst_labels, batch_src_labels = load_model_input(feat_data_num,
                                                                                                          feat_data_cat,
                                                                                                          feat_data_label,
                                                                                                          dst_nodes,
                                                                                                          src_nodes,
                                                                                                          blocks=blocks,
                                                                                                          device=args[
                                                                                                              'device'])
                blocks = [block.to(args['device']) for block in blocks]
                val_batch_logits, _ = model(blocks, batch_num_inputs, batch_src_labels, batch_cat_inputs)

                mask = batch_dst_labels == 2
                val_batch_logits = val_batch_logits[~mask]
                batch_dst_labels = batch_dst_labels[~mask]

                if batch_dst_labels.shape[0] == 0:
                    continue

                val_loss_list += loss_fn(val_batch_logits, batch_dst_labels)
                val_batch_pred = torch.sum(torch.argmax(val_batch_logits.clone().detach(), dim=1) == batch_dst_labels) / \
                                 batch_dst_labels.shape[0]
                val_acc_list = val_acc_list + val_batch_pred * batch_dst_labels.shape[0]
                val_all_list = val_all_list + batch_dst_labels.shape[0]

                if step % 100 == 0:
                    score = torch.softmax(val_batch_logits.clone().detach(), dim=1)[:, 1].cpu().numpy()
                    try:

                        print('In epoch:{:03d}|batch:{:04d}, val_loss:{:4f}, val_ap:{:.4f}, '
                              'val_acc:{:.4f}, val_auc:{:.4f}'.format(epoch, step,
                                                                      val_loss_list / val_all_list,
                                                                      average_precision_score(
                                                                          batch_dst_labels.cpu().numpy(), score),
                                                                      val_batch_pred.detach(),
                                                                      roc_auc_score(batch_dst_labels.cpu().numpy(),
                                                                                    score)))

                    except:
                        pass

        earlystoper.earlystop(val_loss_list / val_all_list, model)
        if earlystoper.is_earlystop:
            print("Early Stopping!")
            break

    # 加载最佳模型
    b_model = earlystoper.best_model.to(args['device'])
    with torch.no_grad():
        for step, (src_nodes, dst_nodes, blocks) in enumerate(valid_dataloader):
            batch_num_inputs, batch_cat_inputs, batch_dst_labels, batch_src_labels = load_model_input(feat_data_num,
                                                                                                      feat_data_cat,
                                                                                                      feat_data_label,
                                                                                                      dst_nodes,
                                                                                                      src_nodes,
                                                                                                      blocks=blocks,
                                                                                                      device=args[
                                                                                                          'device'])
            blocks = [block.to(args['device']) for block in blocks]
            val_batch_logits, _ = b_model(blocks, batch_num_inputs, batch_src_labels, batch_cat_inputs)
            test_predictions[dst_nodes] = val_batch_logits

    test_score = torch.softmax(test_predictions, dim=1)[valid_idx, 1].detach().cpu().numpy()  # 1的部分
    y_target = feat_data_label[valid_idx].detach().cpu().numpy()
    test_score1 = torch.argmax(test_predictions, dim=1)[valid_idx].detach().cpu().numpy()

    mask = y_target != 2
    test_score = test_score[mask]
    y_target = y_target[mask]
    test_score1 = test_score1[mask]

    print("test AUC:", roc_auc_score(y_target, test_score))
    print("test f1:", f1_score(y_target, test_score1, average="macro"))
    print("test AP:", average_precision_score(y_target, test_score))