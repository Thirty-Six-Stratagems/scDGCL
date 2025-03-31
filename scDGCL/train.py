import os
import time
import numpy as np
import torch
from utils import adjust_learning_rate, graph_cluster_embedding, get_device
from scDGCL import GCCL, DCL, DataAug, Encoder, Projector, Cellconloss, Clusterconloss
from GGM import construct_att_graph, generateAdj, combine_graphs



def fill_nan_embedding(embedding):
    return np.nan_to_num(embedding, nan=0.0)


def run(gene_exp, cluster_number, dataset, real_label, epochs, lr, temperature, dropout,
        batch_size, m,
        save_pred=True, noise=None, use_cpu=None, cluster_methods=None, th_rate=None, attn_rate=None,
        prob_feature=None, prob_edge=None
        ):
    if cluster_methods is None:
        cluster_methods = []
    final_results = {}

    start = time.time()
    final_feature, real_label = train_model(
        dataset_name=dataset,
        gene_exp=gene_exp,
        cluster_number=cluster_number,
        real_label=real_label,
        epochs=epochs, lr=lr,
        temperature=temperature,
        dropout=dropout,
        batch_size=batch_size,
        m=m, save_pred=save_pred, noise=noise,
        use_cpu=use_cpu,
        th_rate=th_rate,
        attn_rate=attn_rate,
        prob_feature=prob_feature,
        prob_edge=prob_edge
    )

    if save_pred:
        final_results[f"features"] = final_feature
    elapsed = time.time() - start
    final_feature = fill_nan_embedding(final_feature)
    final_res_eval = graph_cluster_embedding(final_feature, cluster_number, real_label, save_pred=save_pred)
    final_results = {**final_results, **final_res_eval, "dataset": dataset}

    return final_results


def train_model(dataset_name, gene_exp, cluster_number, real_label, epochs, lr,
                temperature, dropout, batch_size, m,
                save_pred=False, noise=None, use_cpu=None, evaluate_training=True, th_rate=None, attn_rate=None,
                prob_feature=None, prob_edge=None):
    device = get_device(use_cpu)
    encoder_dims = [gene_exp.shape[1], int(gene_exp.shape[1] * 0.9), 256, 128]
    data_aug_model = DataAug(dropout=dropout)

    encoder_1 = Encoder(encoder_dims)
    encoder_2 = Encoder(encoder_dims)
    cell_projector = Projector(128, 256, 128)
    cluster_projector = Projector(128, 64, cluster_number)
    dcl_model = DCL(encoder_1, encoder_2, cell_projector, cluster_projector, cluster_number, m=m)
    data_aug_model.to(device)
    dcl_model.to(device)
    dcl_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, dcl_model.parameters()), lr=lr)
    criterion_instance = Cellconloss(temperature=temperature)
    criterion_cluster = Clusterconloss(cluster_number, temperature=temperature)

    gccl_model = GCCL(input_dim=128, gcn_dim=64, projector_dim=cluster_number, prob_feature=prob_feature,
                      prob_edge=prob_edge,
                      tau=0.8, alpha=0.55,
                      beta=0.4, dropout=0.2).to(device)
    gccl_optimizer = torch.optim.Adam(gccl_model.parameters(), lr=0.001)
    adj_1, _, _ = generateAdj(gene_exp, k=10, th_rate=th_rate)

    max_value = -1
    best_adj = None
    idx = np.arange(len(gene_exp))

    def save_best_model(state, dataset_name=dataset_name, filename='best_model.pth.tar'):
        save_dir = os.path.join(os.getcwd(), 'save_model', dataset_name)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, filename)
        torch.save(state, file_path)

    for epoch in range(epochs):
        dcl_model.train()
        adjust_learning_rate(dcl_optimizer, epoch, lr)
        np.random.shuffle(idx)
        loss_instance_ = 0
        loss_cluster_ = 0

        for pre_index in range(len(gene_exp) // batch_size + 1):
            c_idx = np.arange(pre_index * batch_size, min(len(gene_exp), (pre_index + 1) * batch_size))
            if len(c_idx) == 0:
                continue
            c_idx = idx[c_idx]
            c_inp = gene_exp[c_idx]

            input1 = data_aug_model(torch.FloatTensor(c_inp))
            input2 = data_aug_model(torch.FloatTensor(c_inp))

            if noise is None or noise == 0:
                input1 = torch.FloatTensor(input1).to(device)
                input2 = torch.FloatTensor(input2).to(device)
            else:
                noise_vec = torch.FloatTensor(np.random.normal(loc=0, scale=noise, size=input1.shape))
                input1 = torch.FloatTensor(input1 + noise_vec).to(device)
                noise_vec = torch.FloatTensor(np.random.normal(loc=0, scale=noise, size=input2.shape))
                input2 = torch.FloatTensor(input2 + noise_vec).to(device)

            cell_level_1, cluster_level_1, cell_level_2, cluster_level_2 = dcl_model(input1, input2)
            features_instance = torch.cat([cell_level_1.unsqueeze(1), cell_level_2.unsqueeze(1)], dim=1)
            features_cluster = torch.cat([cluster_level_1.t().unsqueeze(1), cluster_level_2.t().unsqueeze(1)], dim=1)

            loss_instance = criterion_instance(features_instance)
            loss_cluster = criterion_cluster(features_cluster)
            dcl_loss = loss_instance + loss_cluster

            loss_instance_ += loss_instance.item()
            loss_cluster_ += loss_cluster.item()

            dcl_optimizer.zero_grad()
            dcl_loss.backward()
            dcl_optimizer.step()

        if evaluate_training and real_label is not None:
            dcl_model.eval()
            with torch.no_grad():
                cell_level_1, _, _, _ = dcl_model(torch.FloatTensor(gene_exp).to(device), None)
                features = cell_level_1.detach()

            best_performance_in_10_epochs = 0
            best_results_in_10_epochs = {'epoch': None,
                                         'gccl_Loss': None,
                                         'ARI': None,
                                         'NMI': None}
            for gccl_epoch in range(10):
                x = torch.tensor(gene_exp, dtype=torch.float32).to(device)
                _, _, adj_2 = construct_att_graph(x, h=5, attn_rate=attn_rate)
                edge_index, edge_weight, adj = combine_graphs(adj_1, adj_2)
                adj = torch.FloatTensor(adj).to(device)

                gccl_model.train()
                gccl_train_feature, x_imp, loss_cl = gccl_model(features, adj)
                mae_f = torch.nn.L1Loss(reduction='mean')
                mask = torch.where(features != 0, torch.ones_like(features),
                                   torch.zeros_like(features))
                loss_mae = mae_f(mask * x_imp, mask * features)
                gccl_loss = loss_mae + 0.8871 * loss_cl
                gccl_optimizer.zero_grad()
                gccl_loss.backward()
                gccl_optimizer.step()
                gccl_model.eval()
                with torch.no_grad():
                    if isinstance(real_label, np.ndarray):
                        real_label = torch.tensor(real_label)
                    real_label = real_label.clone().detach().to(device).long()
                    gccl_val_feature, _, _ = gccl_model(features, adj)
                    gccl_val_feature = gccl_val_feature.detach().cpu().numpy()
                real_label = real_label.cpu()

                gccl_val_feature = fill_nan_embedding(gccl_val_feature)
                gccl_res = graph_cluster_embedding(gccl_val_feature, cluster_number, real_label, save_pred=save_pred)
                current_performance = gccl_res['ari'] + gccl_res['nmi']
                if current_performance > max_value:
                    max_value = current_performance
                    best_adj = adj
                    save_best_model({
                        'epoch': epoch,
                        "best_adj": best_adj,
                        'dcl_state_dict': dcl_model.state_dict(),
                        'gccl_state_dict': gccl_model.state_dict(),
                        'dcl_optimizer': dcl_optimizer.state_dict(),
                        'gccl_optimizer': gccl_optimizer.state_dict(),
                        'max_value': max_value,
                    })
                if current_performance > best_performance_in_10_epochs:
                    best_performance_in_10_epochs = current_performance
                    best_results_in_10_epochs = {
                        'epoch': gccl_epoch,
                        'gccl_Loss': gccl_loss.item(),
                        'ARI': gccl_res['ari'],
                        'NMI': gccl_res['nmi']
                    }
                if gccl_epoch == 9:
                    print(
                        f"Epoch {epoch}: DCL Loss: {dcl_loss.item()}, ",
                        f"GCCL Loss: {best_results_in_10_epochs['gccl_Loss']}, "
                        f"ARI: {best_results_in_10_epochs['ARI']}, NMI: {best_results_in_10_epochs['NMI']}")

    best_model_path = os.path.join(os.getcwd(), 'save_model', dataset_name, 'best_model.pth.tar')
    if os.path.isfile(best_model_path):
        print(f"=> loading best model '{best_model_path}'")
        checkpoint = torch.load(best_model_path)
        val_adj = checkpoint["best_adj"]
        dcl_model.load_state_dict(checkpoint['dcl_state_dict'])
        gccl_model.load_state_dict(checkpoint['gccl_state_dict'])
        print(f"=> loaded best model (epoch {checkpoint['epoch']})")
    else:
        print(f"=> no best model found at '{best_model_path}'")

    dcl_model.eval()
    gccl_model.eval()
    with torch.no_grad():
        cell_level_1, _, _, _ = dcl_model(torch.FloatTensor(gene_exp).to(device), None)
        features = cell_level_1.detach()
        final_features, _, _ = gccl_model(features, val_adj)
        final_features = final_features.detach().cpu().numpy()

    return final_features, real_label
