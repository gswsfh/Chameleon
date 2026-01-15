
import datetime
import math
import random

import Levenshtein
import scipy
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
import logging
import pickle
import awkward as ak
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv, global_mean_pool
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn

import sbemEmbed

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

class TrafficMetadataDataset(Dataset):
    FLOWMAXLENGTH=32
    PAD_VAL=0
    TST=0.1
    def __init__(self,filepath,cache_dir="./concache", random_seed=42,sample=False,sample_num=10*10000):
        super(Dataset, self).__init__()
        np.random.seed(random_seed)
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_path_pre=os.path.join(self.cache_dir,filepath.split("/")[-2])

        feafilepath=self.cache_path_pre+"-fea.pkl"
        servergrpidfilepath=self.cache_path_pre+"-server-grpid.pkl"
        servicegrpidfilepath=self.cache_path_pre+"-service-grpid.pkl"
        idsfilepath=self.cache_path_pre+"-ids.pkl"
        if os.path.exists(feafilepath):
            
            with open(feafilepath, 'rb') as f:
                self.fea = pickle.load(f)
            with open(servergrpidfilepath, 'rb') as f:
                self.ip_gid = pickle.load(f)
            with open(servicegrpidfilepath, 'rb') as f:
                self.port_gid = pickle.load(f)
            with open(idsfilepath, 'rb') as f:
                self.ids = pickle.load(f)
        else:
           
            self.fea,self.ip_gid,self.port_gid,self.ids=self._build_flow_feas(filepath,sample,sample_num)
            with open(feafilepath, 'wb') as f:
                pickle.dump(self.fea,f)
            with open(servergrpidfilepath, 'wb') as f:
                pickle.dump(self.ip_gid,f)
            with open(servicegrpidfilepath, 'wb') as f:
                pickle.dump(self.port_gid,f)
            with open(idsfilepath, 'wb') as f:
                pickle.dump(self.ids,f)

       

    def _build_flow_feas(self,filename,issampled=False,sampleNum=10*10000,PAD_VAL = 1,FLOWMAXLENGTH=32):
       
        tbl = ak.from_parquet(filename)
        mask = ak.num(tbl["pktsize"]) > 1
        tbl = tbl[mask]
       
        if issampled:
           
            total = len(tbl)
            sample_size = min(sampleNum, total) 
            pick_idx = np.random.choice(total, size=sample_size, replace=False)
            tbl = tbl[pick_idx]
            
       
        pkts = ak.values_astype(ak.fill_none(ak.pad_none(tbl["pktsize"], FLOWMAXLENGTH, clip=True), PAD_VAL), "int32")
        dirs = ak.values_astype(ak.fill_none(ak.pad_none(tbl["pktdirs"], FLOWMAXLENGTH, clip=True), 0), "float32")
       
        X_pkt = ak.to_numpy(pkts).astype(np.float32)    # (N, 128)
        X_pkt = (2 / np.pi) * np.arctan(np.log2(X_pkt)/8)
        X_dir = dirs.to_numpy().astype(np.float32)
       
        X = np.stack([X_pkt, X_dir],axis=1).astype(np.float32)
      
        ids_np = np.array([
            tuple(row.values()) for row in ak.to_list(tbl[["srcip", "dstip", "srcport", "dstport", "proto"]])
        ], dtype=[
            ('srcip', 'U15'),
            ('dstip', 'U15'),
            ('srcport', 'i4'),
            ('dstport', 'i4'),
            ('proto', 'i4')
        ])
    
  
        server_inner=dict()
        for l_i,line in tqdm(enumerate(ids_np)):
            srcip,dstip,srcport,dstport,proto=line
            if srcport<dstport:
                serverid=(srcip,srcport)
            else:
                serverid=(dstip,dstport)
            if serverid not in server_inner:
                server_inner[serverid]=list()
            server_inner[serverid].append(l_i)
   
        X_server_grp=self._sample_grps(server_inner)
  
        service_inner=dict()
        for l_i,line in tqdm(enumerate(ids_np)):
            srcip,dstip,srcport,dstport,proto=line
            if srcport<dstport:
                serverid=srcport
            else:
                serverid=dstport
            if serverid not in service_inner:
                service_inner[serverid]=list()
            service_inner[serverid].append(l_i)
        X_service_grp=self._sample_grps(service_inner)

        return X[:,0,:],X_server_grp,X_service_grp,ids_np

    def _build_task_samples(self,x,maskrate=0.3,maskvalue=-0.1):

        x_len=np.argmax(x[:,1] == 0,axis=1)
        x_len[x_len==0]=32
   
        X_mask=x[:,0].copy()
        mask_len=(x_len*maskrate).astype(np.int8)
        mask_len[mask_len==0]=1
        for i,(L,k) in enumerate(zip(x_len,mask_len)):
            mask_indices=np.random.choice(L, size=k, replace=False)
            X_mask[i,mask_indices] = maskvalue
   
     
        X_out_of_order=x[:,0].copy()
        for i,(L,k) in enumerate(zip(x_len,mask_len)):
            mask_indices=np.random.choice(L, size=k, replace=False)
            mask_indices[mask_indices==0]=1
            for item in sorted(mask_indices, reverse=True):
                X_out_of_order[i,item], X_out_of_order[i,item-1] = X_out_of_order[i,item-1], X_out_of_order[i,item]
     
       
        X_missing=x[:,0].copy()
        for i,(L,k) in enumerate(zip(x_len,mask_len)):
            mask_indices=np.random.choice(L, size=k, replace=False)
            data_deleted=np.delete(X_missing[i],mask_indices)
            X_missing[i] = np.pad(data_deleted, (0, k), constant_values=0)
  
       
        X_duplicate=x[:,0].copy()
        for i,(L,k) in enumerate(zip(x_len,mask_len)):
            mask_indices=np.random.choice(L, size=k, replace=False)
            data_dup=X_duplicate[i,mask_indices]
            data_dup_seq = np.concatenate([
                X_duplicate[i,:L],
                data_dup
            ])[:self.FLOWMAXLENGTH]
            if len(data_dup_seq)<self.FLOWMAXLENGTH:
                X_duplicate[i] = np.pad(data_dup_seq, (0, self.FLOWMAXLENGTH-len(data_dup_seq)), constant_values=0)
            else:
                X_duplicate[i]=data_dup_seq
   
        X_task=np.stack([X_mask,X_out_of_order,X_duplicate,X_missing],axis=1)
        return X_task

    def _sample_grps(self,grpsids):
    
        res=list()
        res_append = res.append
        randrange = random.randrange
        for id,indexs in tqdm(grpsids.items()):
            n = len(indexs)
            if n==1:
                x = indexs[0]
                res_append([x, x, x])
            elif len(indexs)==2:
                a, b = indexs
                res_append([a, b, b])
                res_append([b, a, a])
            else:
                for i,item in enumerate(indexs):
                    j = randrange(n)
                    while j == i:
                        j = randrange(n)
                    k = randrange(n)
                    while k == i or k == j:
                        k = randrange(n)
                    res_append([item, indexs[j], indexs[k]])
        return res

    def __getitem__(self, index):
        ip_index=self.ip_gid[index]
        port_index=self.port_gid[index]
        ip_data=self.fea[ip_index]
        port_data=self.fea[port_index]
        return ip_data,port_data,index

    def get_id(self,index):
        return self.ids[index]

    def get_ids(self,indexs):
        return self.ids[indexs]

    def __len__(self) -> int:
        return len(self.fea)

class TrafficMetadataGraphDataset(Dataset):
    FLOWMAXLENGTH=32
    PAD_VAL=0
    TST=0.1

    def __init__(self,filepath,cache_dir="./con-graph-cache", random_seed=42,sample=False,sample_num=10_0000,graphsize=1_0000):
        super(Dataset, self).__init__()
        np.random.seed(random_seed)
        self.grahsize=graphsize
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_path_pre=os.path.join(self.cache_dir,filepath.split("/")[-2])
        groupdatafilepath=self.cache_path_pre+"-graphdata.pkl"
        if os.path.exists(groupdatafilepath):
           
            with open(groupdatafilepath, 'rb') as f:
                self.graphdata = pickle.load(f)
        else:
          
            self.graphdata=self._build_flow_feas(filepath,sample,sample_num)
            with open(groupdatafilepath, 'wb') as f:
                pickle.dump(self.graphdata,f)

      

    def _embedding_flows(self,flows):
 
        emb_modelpath="embed.pt"
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data_fea=flows.to(torch.float32).to(device)
        emb_network=sbemEmbed.Network(sbemEmbed.TrafficMetadataDataset.FLOWMAXLENGTH).to(device)
        emb_network.load_state_dict(torch.load(emb_modelpath))
        emb_network.eval()
        _,data_embed=emb_network(data_fea)
        return data_embed


    def _build_flow_feas(self,filename,issampled=False,sampleNum=10*10000,PAD_VAL = 1,FLOWMAXLENGTH=32):
    
        tbl = ak.from_parquet(filename)
        mask = ak.num(tbl["pktsize"]) > 1
        tbl = tbl[mask]
   
        if issampled:
           
            total = len(tbl)
            sample_size = min(sampleNum, total)  
            pick_idx = np.random.choice(total, size=sample_size, replace=False)
            tbl = tbl[pick_idx]
           
    
        pkts = ak.values_astype(ak.fill_none(ak.pad_none(tbl["pktsize"], FLOWMAXLENGTH, clip=True), PAD_VAL), "int32")
        
        X_pkt = ak.to_numpy(pkts).astype(np.float32)    # (N, 128)
        X_pkt = (2 / np.pi) * np.arctan(np.log2(X_pkt)/8)
     
        X = X_pkt.astype(np.float32)
    
        X_np = X.data
        X_t  = torch.from_numpy(X_np).float()
        X_embed = self._embedding_flows(X_t).cpu().detach().numpy()
       
        X_ids = np.array([
            tuple(row.values()) for row in ak.to_list(tbl[["srcip", "dstip", "srcport", "dstport", "proto"]])
        ], dtype=[
            ('srcip', 'U15'),
            ('dstip', 'U15'),
            ('srcport', 'i4'),
            ('dstport', 'i4'),
            ('proto', 'i4')
        ])

     
     
        X_len=len(X)
        self.graphdata=list()
        for g_index in range(0,X_len,self.grahsize):
            g_index_min=g_index
            g_index_max=min(g_index+self.grahsize,X_len)
            X_win=X_embed[g_index_min:g_index_max]
            X_oral_win=X[g_index_min:g_index_max]
            X_ids_win=X_ids[g_index_min:g_index_max]
            graph,graphnodeids=self._build_graph_from_flows(X_ids_win,X_win)
       
            ip_inner=dict()
            for l_i,line in tqdm(enumerate(X_ids_win)):
                srcip,dstip,srcport,dstport,proto=line
                if srcport<dstport:
                    serverid=(srcip,srcport)
                else:
                    serverid=(dstip,dstport)
                if serverid not in ip_inner:
                    ip_inner[serverid]=list()
                ip_inner[serverid].append(l_i)
           
            X_ip_grp=self._sample_grps(ip_inner)
            X_ip_grp_oral=X_oral_win[X_ip_grp]
       
            port_inner=dict()
            for l_i,line in tqdm(enumerate(X_ids_win)):
                srcip,dstip,srcport,dstport,proto=line
                if srcport<dstport:
                    serverid=srcport
                else:
                    serverid=dstport
                if serverid not in port_inner:
                    port_inner[serverid]=list()
                port_inner[serverid].append(l_i)
            X_port_grp=self._sample_grps(port_inner)
            X_port_grp_oral=X_oral_win[X_ip_grp]
            self.graphdata.append([graph,X_ip_grp, X_port_grp,X_ip_grp_oral,X_port_grp_oral])

       
        return self.graphdata


    def _sample_grps(self,grpsids):
       
        res=list()
        res_append = res.append
        randrange = random.randrange
        for id,indexs in tqdm(grpsids.items()):
            n = len(indexs)
            if n==1:
                x = indexs[0]
                res_append([x, x, x])
            elif len(indexs)==2:
                a, b = indexs
                res_append([a, b, b])
                res_append([b, a, a])
            else:
                for i,item in enumerate(indexs):
                    j = randrange(n)
                    while j == i:
                        j = randrange(n)
                    k = randrange(n)
                    while k == i or k == j:
                        k = randrange(n)
                    res_append([item, indexs[j], indexs[k]])
        return res

    def __getitem__(self, index):
        return self.graphdata[index],index

    def __len__(self) -> int:
        return len(self.graphdata)

    def _build_graph_from_flows(self,flowids,flowfeas):
        ip_to_ids=dict()
        for flowid in flowids:
            src,dst=flowid[0],flowid[1]
            for ip in [src,dst]:
                if ip not in ip_to_ids:
                    ip_to_ids[ip]=len(ip_to_ids)

        srcids=list()
        dstids=list()
        for flowid,feat in zip(flowids,flowfeas):
            srcids.append(ip_to_ids[flowid[0]])
            dstids.append(ip_to_ids[flowid[1]])

        edge_index = torch.tensor([srcids, dstids], dtype=torch.long)
        edge_attr = torch.tensor(flowfeas, dtype=torch.float)
        num_nodes = len(ip_to_ids)
        # node_ids = torch.arange(num_nodes, dtype=torch.long)

        return Data(
            x=torch.zeros(num_nodes, 1) ,
            edge_index=edge_index,
            edge_attr=edge_attr
        ), ip_to_ids

class Network(nn.Module):
    def __init__(self, feature_size=128, latent_dim=128):
        super(Network, self).__init__()
        self.ln=nn.Linear(feature_size, latent_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        x=self.relu(self.ln(x))
        return x

class GAT(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, edge_dim, heads=4, dropout=0.2):
        super(GAT, self).__init__()
        self.conv1 = GATv2Conv(in_dim, hidden, heads=heads, edge_dim=edge_dim,
                               dropout=dropout, concat=True)
        self.conv2 = GATv2Conv(heads*hidden, hidden, heads=1, edge_dim=edge_dim,
                               dropout=dropout, concat=False)
        self.edge_projector = nn.Sequential(
            nn.Linear(hidden * 2 + edge_dim, hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, out_dim)
        )
    def forward(self, x, edge_index, edge_attr, batch=None):
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        src, dst = edge_index
        edge_repr = torch.cat([
            x[src],      # [E, hidden]
            x[dst],      # [E, hidden]
            edge_attr    # [E, edge_dim]
        ], dim=-1)       # [E, 2*hidden + edge_dim]

       
        edge_embeddings = self.edge_projector(edge_repr)  # [E, edge_embed_dim]
        return F.normalize(edge_embeddings, dim=1)

def collate_fn(batch):
    graphs = []
    ip_grps = []
    port_grps = []
    ip_orals = []
    port_orals = []
    indices = []

    for item, idx in batch:
        graph, ip_grp, port_grp, ip_oral, port_oral = item
        graphs.append(graph)
        ip_grps.append(ip_grp)        # numpy array [N, 3, 128]
        port_grps.append(port_grp)    # numpy array [N, 3, 128]
        ip_orals.append(ip_oral)      # numpy array [N, 3, 32]
        port_orals.append(port_oral)  # numpy array [N, 3, 32]
        indices.append(idx)

 
    batched_graph = Batch.from_data_list(graphs)


    return (
        batched_graph,
        ip_grps,
        port_grps,
        ip_orals,
        port_orals,
        indices
    )


def calSimBetweenFlow(flow1,flow2, temperature=0.1):
  
  
    mask=(flow1==0)&(flow2==0)
    flow1=flow1[~mask]
    flow2=flow2[~mask]
    def calCosSim(flow1,flow2):
   
        if isinstance(flow1, list):
            flow1=np.array(flow1)
        if isinstance(flow2, list):
            flow2=np.array(flow2)
        cos_sim = flow1.dot(flow2) / (np.linalg.norm(flow1) * np.linalg.norm(flow2))
        return cos_sim
    def calJaroWinkler(flow1,flow2):
        sim = Levenshtein.jaro_winkler(flow1 , flow2 )
        return sim
    def calWasserstein(flow1,flow2):
        sim=1-scipy.stats.wasserstein_distance(flow1,flow2)
        return sim
    sims =[calCosSim(flow1,flow2),calJaroWinkler(flow1,flow2),calWasserstein(flow1,flow2)]
    max_sim = max(sims)
    exp_vals = [math.exp((s - max_sim) / temperature) for s in sims]
    sum_exp = sum(exp_vals)
    weights = [e / sum_exp for e in exp_vals]
    fused = sum(w * s for w, s in zip(weights, sims))
    return fused

def judgeSim(flow,delta=0.75):
  
    sims=[[calSimBetweenFlow(item[0],item[1])>delta,calSimBetweenFlow(item[0],item[2])>delta] for item in flow]
    return torch.tensor(sims)

def manual_triplet_loss_mse(
        anchor: torch.Tensor,
        sample1: torch.Tensor,
        sample2: torch.Tensor,
        pair_labels: torch.Tensor,   # (B, 2), 1=positive, 0=negative
        # margin: float = 0.2,
        weight_pospos: float = 1.0,
        weight_negneg: float = 1.0,
        reduction: str = 'mean'
):
    
    def mse_distance(x, y):
        return F.mse_loss(x, y, reduction='none').mean(dim=-1)  # (B,)
    d1 = mse_distance(anchor, sample1)  # (B,)
    d2 = mse_distance(anchor, sample2)  # (B,)

    B = anchor.shape[0]
    total_loss = torch.zeros(B, device=anchor.device)

   
    mask_10 = (pair_labels[:, 0] == 1) & (pair_labels[:, 1] == 0)
    if mask_10.any():
  
        total_loss[mask_10] = d1[mask_10] - d2[mask_10]

    mask_01 = (pair_labels[:, 0] == 0) & (pair_labels[:, 1] == 1)
    if mask_01.any():
        total_loss[mask_01] = d2[mask_01] - d1[mask_01]

 
    # mask_11 = (pair_labels[:, 0] == 1) & (pair_labels[:, 1] == 1)
    # if mask_11.any():
    #     # loss_11 = torch.abs(d1[mask_11] - d2[mask_11])
    #     total_loss[mask_11] = weight_pospos * loss_11
    #
    # # ---- (0, 0): both negative → push both far
    # mask_00 = (pair_labels[:, 0] == 0) & (pair_labels[:, 1] == 0)
    # if mask_00.any():
    #     total_loss[mask_00] = weight_negneg * (-d1[mask_00] - d2[mask_00])


    if reduction == 'mean':
        return total_loss.mean()
    elif reduction == 'none':
        return total_loss
    else:
        raise ValueError("reduction must be 'mean' or 'none'")


if __name__ == '__main__':
    learning_rate=0.0001
    epochsize=20
    batchsize=16
    step=0 
    modelpath="conn.pt"
    graphpath="graph.pt"
    emb_modelpath="embed.pt"
    sampled=True
    samplenum=10_0000
    graphsize=1_0000
    trainrate=0.7
    now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    filepath=""
    rawdataset=TrafficMetadataGraphDataset(filepath,sample=True,sample_num=samplenum,graphsize=graphsize)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network=Network(TrafficMetadataDataset.FLOWMAXLENGTH).to(device)
    graphnet=GAT(in_dim=1, hidden=128, out_dim=32, edge_dim=128)
    criterion_con=manual_triplet_loss_mse
    optimizer=torch.optim.Adam(list(network.parameters()) + list(graphnet.parameters()),lr=learning_rate)

    train_size = int(trainrate * len(rawdataset))
    test_size = len(rawdataset) - train_size
    traindataset,testdataset=random_split(rawdataset,[train_size,test_size])
    train_dataloader = DataLoader(dataset=traindataset, batch_size=batchsize, shuffle=True,collate_fn=collate_fn )
    test_dataloader = DataLoader(dataset=testdataset, batch_size=batchsize, shuffle=False,collate_fn=collate_fn )

    if step==0:
        writer = SummaryWriter(log_dir='./runs')
        network.train()
        for epoch in range(0,epochsize):
            network.train()
            epoch_loss=list()

            for idx,(batched_graph, ip_grps, port_grps, ip_orals, port_orals, indices) in enumerate(train_dataloader,0):
                g_embed = graphnet(
                    batched_graph.x,
                    batched_graph.edge_index,
                    batched_graph.edge_attr,
                    batch=batched_graph.batch 
                )
                g_embed=g_embed.view(-1, rawdataset.grahsize, rawdataset.FLOWMAXLENGTH)
                g_size=g_embed.size(0)
          
                loss=0
            
                for g_i in range(g_size):
                    g_data=g_embed[g_i]
                
                    g_ip_grp=ip_grps[g_i]
                
                    embed_ip=g_data[g_ip_grp]
                  
                    g_ip_oral=ip_orals[g_i]
                    simJudges1=judgeSim(g_ip_oral)
                    loss+=criterion_con(embed_ip[:,0,:],embed_ip[:,1,:],embed_ip[:,2,:],simJudges1)
                  
                    g_port_grp=port_grps[g_i]
                    embed_port=g_data[g_port_grp]
                    g_port_oral=port_orals[g_i]
                    simJudges2=judgeSim(g_port_oral)
                    loss+=criterion_con(embed_port[:,0,:],embed_port[:,1,:],embed_port[:,2,:],simJudges2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())
            writer.add_scalar(now_str+'Training Loss', np.mean(epoch_loss).item(),epoch)
            print(f"epoch={epoch}/{epochsize},loss={np.mean(epoch_loss)}")
            torch.save(network.state_dict(),modelpath)
            torch.save(graphnet.state_dict(),graphpath)
    elif step==1:
        data_embeds=list()
        data_ids=list()
        network.load_state_dict(torch.load(modelpath))
        network.eval()
        graphnet.load_state_dict(torch.load(graphpath))
        graphnet.eval()
        for idx,(batched_graph, ip_grps, port_grps, ip_orals, port_orals, indices) in enumerate(test_dataloader,0):
            g_embed = graphnet(
                batched_graph.x,
                batched_graph.edge_index,
                batched_graph.edge_attr,
                batch=batched_graph.batch 
            )
            g_embed=g_embed.view(-1, rawdataset.grahsize, rawdataset.FLOWMAXLENGTH)


