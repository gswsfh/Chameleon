import json
import logging
import math
import os
import pickle
import random
from collections import Counter

import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader, random_split
import awkward as ak
from tqdm import tqdm
from scipy.stats import entropy

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

class TrafficMetadataGroupDataset(Dataset):
    FLOWMAXLENGTH=32
    PAD_VAL=0
    TST=0.1

    def __init__(self,filepath, cache_dir="./cache", random_seed=42):
        super(Dataset, self).__init__()
        np.random.seed(random_seed)
        self.filepath=filepath
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        base_name = os.path.basename(filepath).rsplit('.', 1)[0]
        self.cache_path = os.path.join(self.cache_dir, f"{base_name}_valid_groups")

        if os.path.exists(self.cache_path+"_data.pkl"):
            
            with open(self.cache_path+"_data.pkl", 'rb') as f:
                self.valid_group_data = pickle.load(f)
            with open(self.cache_path+"_labels.pkl", 'rb') as f:
                self.valid_group_labels = pickle.load(f)
        else:
           
            X,y,flowid=self._build_flow_feas(filepath)
            self._build_and_cache_groups(X,y,flowid)

        self.g_lengths = [len(g) for g in self.valid_group_labels]
        self.cum_lengths = np.cumsum([0] + self.g_lengths)
        self.total_samples = self.cum_lengths[-1]
        

    def _build_flow_feas(self,filename,issampled=False,sampleNum=10*10000,PAD_VAL = 1,FLOWMAXLENGTH=32,MINTS=0.1):
       
        tbl = ak.from_parquet(filename)
        
        if issampled:
            
            total = len(tbl)
            sample_size = min(sampleNum, total)  
            pick_idx = np.random.choice(total, size=sample_size, replace=False)
            # pick_idx = np.sort(pick_idx)
            tbl = tbl[pick_idx]
           
       
        pkts = ak.values_astype(ak.fill_none(ak.pad_none(tbl["pktsize"], FLOWMAXLENGTH, clip=True), PAD_VAL), "int32")
        dirs = ak.values_astype(ak.fill_none(ak.pad_none(tbl["pktdirs"], FLOWMAXLENGTH, clip=True), 0), "float32")
        timestamps = ak.values_astype(tbl["pktts"], "float32")
        time_diffs = timestamps[:, 1:] - timestamps[:, :-1]  # (N, L-1)
        time_deltas = ak.values_astype(time_diffs, "float32")
        time_deltas = ak.values_astype(ak.fill_none(ak.pad_none(time_deltas, FLOWMAXLENGTH, clip=True), 0), "float32")
  
        X_pkt = ak.to_numpy(pkts).astype(np.float32)    # (N, 128)
        X_pkt = (2 / np.pi) * np.arctan(np.log2(X_pkt)/8)
        X_dir = ak.to_numpy(ak.where(dirs == -1, 0.5, dirs)).astype(np.float32)
        X_time=  ak.to_numpy(time_deltas)
        first_ts = np.log1p(timestamps[:, 0:1]-np.min(timestamps))/12
        mask = X_time < MINTS
        X_time[mask] = 0.0
        X_time[~mask] = np.round(X_time[~mask] / 0.05) * 0.05
        X_time = np.clip(X_time, 0.0, 10.0)/10
        X_time=X_time+first_ts.to_numpy()
        
        X = np.stack([X_pkt, X_dir,X_time],axis=1).astype(np.float32)
        y = (tbl["label"] != "Benign").to_numpy().astype(np.int64)
       
        return X,y,ak.to_dataframe(tbl[["srcip",'dstip','srcport','dstport','proto']])

    def _build_and_cache_groups(self,flowfea,flowlabel,flowids):
        
        
        data_cs = self._assign_client_server(flowids)
        
        grouped = data_cs.groupby(['server_ip', 'server_port'])
        self.valid_group_data = list()
        self.valid_group_labels = list()
        for g_name, g_data_indexs in tqdm(grouped):
            if len(g_data_indexs) > 2:
               
                g_small = flowfea[g_data_indexs.index,:]
                g_labels=flowlabel[g_data_indexs.index]
                self.valid_group_data.append(g_small)
                self.valid_group_labels.append(g_labels)
        del data_cs, grouped
        
        with open(self.cache_path+"_data.pkl", 'wb') as f:
            pickle.dump(self.valid_group_data, f)
        with open(self.cache_path+"_labels.pkl", 'wb') as f:
            pickle.dump(self.valid_group_labels, f)

    def _assign_client_server(self,df):
     
        cond = df['srcport'] < df['dstport']
        df['client_ip'] = np.where(cond, df['dstip'], df['srcip'])
        df['server_ip'] = np.where(cond, df['srcip'], df['dstip'])
        df['client_port'] = np.where(cond, df['dstport'], df['srcport'])
        df['server_port'] = np.where(cond, df['srcport'], df['dstport'])
        return df

    def _group_by_server_and_port(self,df):
        dataGrp=df.groupby(['server_ip',"server_port"])
        return dataGrp

    def _sample_triplet_from_array(self, g_array, anchor_idx):
        
        N = g_array.shape[0]
        i = anchor_idx

        
        idx1 = np.random.randint(0, N - 1)
        if idx1 >= i:
            idx1 += 1

        
        idx2 = np.random.randint(0, N - 2)
        if idx2 >= min(i, idx1):
            idx2 += 1
        if idx2 >= max(i, idx1):
            idx2 += 1

        return g_array[[i, idx1, idx2], :]  # (3, 3)

    def __getitem__(self, index):
        if index < 0 or index >= self.total_samples:
            raise IndexError("Index out of range")
       
        group_idx = np.searchsorted(self.cum_lengths, index, side='right') - 1
        intra_idx = index - self.cum_lengths[group_idx] 
        g_array = self.valid_group_data[group_idx]
        g_label=self.valid_group_labels[group_idx]
        triplet = self._sample_triplet_from_array(g_array, intra_idx)
        label=g_label[intra_idx]
        return triplet.filled(0),label

    def __len__(self) -> int:
        return self.total_samples

class TrafficMetadataDataset(Dataset):
    FLOWMAXLENGTH=32
    PAD_VAL=0
    TST=0.1
    def __init__(self,filepath,cache_dir="./cache", random_seed=42,sample=False,sample_num=10*10000):
        super(Dataset, self).__init__()
        np.random.seed(random_seed)
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_path_pre=os.path.join(self.cache_dir,filepath.split("/")[-2])
        if os.path.exists(self.cache_path_pre+"-fea.pkl"):
            with open(self.cache_path_pre+"-fea.pkl", 'rb') as f:
                self.fea = pickle.load(f)
            with open(self.cache_path_pre+"-label.pkl", 'rb') as f:
                self.label = pickle.load(f)
            with open(self.cache_path_pre+"-ids.pkl", 'rb') as f:
                self.ids = pickle.load(f)
            if sample:
                rng = np.random.default_rng(seed=42)  
                sampled_indices = rng.choice(len(self.label), size=sample_num, replace=False)
                self.fea = self.fea[sampled_indices]
                self.label = self.label[sampled_indices]
                self.ids = self.ids[sampled_indices]
        else:
            self.fea,self.label,self.ids=self._build_flow_feas(filepath,sample,sample_num)
            self.ids = self.ids.values
            with open(self.cache_path_pre+"-fea.pkl", 'wb') as f:
                pickle.dump(self.fea,f)
            with open(self.cache_path_pre+"-label.pkl", 'wb') as f:
                pickle.dump(self.label,f)
            with open(self.cache_path_pre+"-ids.pkl", 'wb') as f:
                pickle.dump(self.ids,f)


    def _build_flow_feas(self,filename,issampled=False,sampleNum=10*10000,PAD_VAL = 1,FLOWMAXLENGTH=32,MINTS=0.1):
        
        tbl = ak.from_parquet(filename)
        
        if issampled:
            
            total = len(tbl)
            sample_size = min(sampleNum, total)  
            pick_idx = np.random.choice(total, size=sample_size, replace=False)
            # pick_idx = np.sort(pick_idx)
            tbl = tbl[pick_idx]
           
        pkts = ak.values_astype(ak.fill_none(ak.pad_none(tbl["pktsize"], FLOWMAXLENGTH, clip=True), PAD_VAL), "int32")
        dirs = ak.values_astype(ak.fill_none(ak.pad_none(tbl["pktdirs"], FLOWMAXLENGTH, clip=True), 0), "float32")
        timestamps = ak.values_astype(tbl["pktts"], "float32")
        time_diffs = timestamps[:, 1:] - timestamps[:, :-1]  # (N, L-1)
        time_deltas = ak.values_astype(time_diffs, "float32")
        time_deltas = ak.values_astype(ak.fill_none(ak.pad_none(time_deltas, FLOWMAXLENGTH, clip=True), 0), "float32")
       
        X_pkt = ak.to_numpy(pkts).astype(np.float32)    # (N, 128)
        X_pkt = (2 / np.pi) * np.arctan(np.log2(X_pkt)/8)
        X_dir = ak.to_numpy(ak.where(dirs == -1, 0.5, dirs)).astype(np.float32)
        X_time=  ak.to_numpy(time_deltas)
        first_ts = np.log1p(timestamps[:, 0:1]-np.min(timestamps))/12
        mask = X_time < MINTS
        X_time[mask] = 0.0
        X_time[~mask] = np.round(X_time[~mask] / 0.05) * 0.05
        X_time = np.clip(X_time, 0.0, 10.0)/10
        X_time=X_time+first_ts.to_numpy()
        
        X = np.stack([X_pkt, X_dir,X_time],axis=1).astype(np.float32)
        y = (tbl["label"] != "Benign").to_numpy().astype(np.int64)
      
        return X,y,ak.to_dataframe(tbl[["srcip",'dstip','srcport','dstport','proto']])

    def __getitem__(self, index):
        return self.fea[index],self.label[index],index

    def get_id(self,index):
        return self.ids[index]

    def get_ids(self,indexs):
        return self.ids[indexs]

    def __len__(self) -> int:
        return len(self.label)

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.en1=nn.Linear(32,16)
        self.relu=nn.ReLU()
        self.en2=nn.Linear(16,8)
        self.de1=nn.Linear(8,16)
        self.de2=nn.Linear(16,32)
    def forward(self, x):
        x=self.relu(self.en1(x))
        x=self.relu(self.en2(x))
        x=self.relu(self.de1(x))
        x=self.de2(x)
        return x

def pca(embeds,labels,size=1):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(embeds)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, alpha=0.7,s=size)
    plt.colorbar(scatter, ticks=[0, 1], label='Label')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA Visualization of 1024 Samples (32D → 2D)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def analysis(data_embeds,data_labels,data_ids):
    
    res=dict()
    for (i_embed,i_label,i_id) in zip(data_embeds,data_labels,data_ids):
        srcip,dstip,srcport,dstport,proto=i_id
        if srcport<dstport:
            serverids=(srcip,srcport)
        else:
            serverids=(dstip,dstport)
        if serverids not in res:
            res[serverids]=list()
        res[serverids].append([i_embed,i_label,i_id])
 
    visisdata=list()
    visislabels=list()
    for serverid,serverdatas in res.items():
        if len(serverdatas)<=2:
            continue
        labels=[item[1] for item in serverdatas]
        if 1 not in labels:
            continue
        visisdata.extend([item[0] for item in serverdatas])
        visislabels.extend([item[1] for item in serverdatas])
        
    model = DBSCAN(eps=0.0001)
    pred = model.fit_predict(visisdata)
    for i_data,i_pred,i_label in zip(visisdata,pred,visislabels):
        print("----------")
        print(i_data)
        print(i_label)
        print(i_pred)
    pca(visisdata,pred,size=10)

def clusterServerdata(filepath,modelpath,sampled=True,samplenum=10_0000):
    batchsize=1024
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_embeds=list()
    data_labels=list()
    data_ids=list()
    data_tss=list()
    network=Network().to(device)
    network.load_state_dict(torch.load(modelpath))
    network.eval()
    rawdataset=TrafficMetadataDataset(filepath,sample=sampled,sample_num=samplenum)
    dataloader = DataLoader(dataset=rawdataset, batch_size=batchsize, shuffle=False)
    for idx,(data_x,data_y,data_index) in enumerate(dataloader,0):
        data_x=data_x.to(torch.float32).to(device)
        data_fea=data_x[:,0,:]*data_x[:,1,:]
        data_ts=data_x[:,2,:]
        output=network(data_fea)
        data_embeds.extend(output.cpu().detach().numpy().tolist())
        data_labels.extend(data_y.tolist())
        data_ids.extend(rawdataset.get_ids(data_index))
        data_tss.extend(data_ts.tolist())
 
    res=dict()
    for (i_embed,i_label,i_id,i_tss) in zip(data_embeds,data_labels,data_ids,data_tss):
        srcip,dstip,srcport,dstport,proto=i_id
        if srcport<dstport:
            serverids=(srcip,srcport)
        else:
            serverids=(dstip,dstport)
        if serverids not in res:
            res[serverids]=list()
        res[serverids].append([i_embed,i_label,i_id,i_tss])

 
    model=DBSCAN(eps=0.1,min_samples=2, algorithm='kd_tree', n_jobs=1)
    data_labels=dict()
    for serverid,serverdata in tqdm(res.items()):
        datas=[item[0] for item in serverdata]
        preds=model.fit_predict(datas).tolist()
        ids=[item[2].tolist() for item in serverdata]
        labels=[item[1] for item in serverdata]
        tss=[item[3] for item in serverdata]
        data_labels[str(serverid)]=[[id,pred,label,ts] for id,pred,label,ts in zip(ids,preds,labels,tss)]
    saveFilepath="./clusters.json"
    with open(saveFilepath,'w') as f:
        json.dump(data_labels,f)

def clientBehaviorInClusters():
    clusterdata="./clusters.json"
    with open(clusterdata,'r') as f:
        data=json.load(f)
    for serverid,serverdatalist in data.items():
        clientbehaviors=dict()
        for id,flowid,flowlabel,flowts in serverdatalist:
            srcip,dstip,srcport,dstport,proto=id
            if srcip in serverid:
                clientip=dstip
            else:
                clientip=srcip
            if clientip not in clientbehaviors:
                clientbehaviors[clientip]=list()
            clientbehaviors[clientip].append([flowid,flowlabel,flowts])
        idlen=[len(item) for _,item in clientbehaviors.items()]
        if len(set(idlen))==1 and idlen[0]==1:
            continue
        labels=[item for _,_,item,_ in serverdatalist]
        if 1 not in labels:
            continue
   
        for id,id_b in clientbehaviors.items():
            id_b.sort(key=lambda x: x[2])
            print(id,[item[:2] for item in id_b])

def clientBehaviorBetwenClusters():
    clusterdata="./clusters.json"
    with open(clusterdata,'r') as f:
        data=json.load(f)
    clientBehavior=dict()
    for serverid,serverdatalist in data.items():
        for id,flowid,flowlabel,flowts in serverdatalist:
            srcip,dstip,srcport,dstport,proto=id
            if srcip in serverid:
                clientip=dstip
            else:
                clientip=srcip
            if clientip not in clientBehavior:
                clientBehavior[clientip]=list()
            clientBehavior[clientip].append([serverid,flowid,flowlabel,flowts])
    
    for clientip,clientdata in clientBehavior.items():
        labels=[item[2] for item in clientdata]
        if 1 not in labels:
            continue
        
        clientdata.sort(key=lambda x: x[3])
        clientdata_tss= np.array([0]+[item[3][0] for item in clientdata])
        clientdata_tss_diff=clientdata_tss[1:]-clientdata_tss[:-1]
        clientdata_tss_all= np.array([item[3] for item in clientdata])
        clientdata_tss_all= np.insert(clientdata_tss_all, 0, 0, axis=1)
        clientdata_tss_all_diff=clientdata_tss_all[:,1:]-clientdata_tss_all[:,:-1]
        feas=[item[:3]+[item_ts_dif.tolist(),item_ts.tolist(),item_tss_all.tolist()] for item,item_ts_dif,item_ts,item_tss_all in zip(clientdata,clientdata_tss_diff,clientdata_tss,clientdata_tss_all_diff)]
        for fea in feas[:100]:
            print(fea)

def train(filepath):
    learning_rate=0.0001
    epochsize=5
    batchsize=1024
    step=1
    modelpath="au.pt"
    sampled=True
    samplenum=10*10000

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network=Network().to(device)
    criterion=nn.MSELoss()
    optimizer=torch.optim.Adam(network.parameters(),lr=learning_rate)


    rawdataset=TrafficMetadataDataset(filepath,sample=sampled,sample_num=samplenum)
    train_size = int(0.7 * len(rawdataset))
    test_size = len(rawdataset) - train_size
    traindataset,testdataset=random_split(rawdataset,[train_size,test_size])
    train_dataloader = DataLoader(dataset=traindataset, batch_size=batchsize, shuffle=True)
    test_dataloader = DataLoader(dataset=testdataset, batch_size=batchsize, shuffle=False)
    if step==0:
        network.train()
        for epoch in range(0,epochsize):
            network.train()
            epoch_loss=list()
            for idx,(data_x,data_y) in enumerate(train_dataloader,0):
                data_x=data_x.to(torch.float32).to(device)
       
                data_fea=data_x[:,0,:]*data_x[:,1,:]
                output=network(data_fea)
                optimizer.zero_grad()
                loss = criterion(data_fea,output)

                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())
  
            print(f"epoch={epoch}/{epochsize},loss={np.mean(epoch_loss)}")
            torch.save(network.state_dict(modelpath), )
    elif step==1:
        data_embeds=list()
        data_labels=list()
        data_ids=list()
        network.load_state_dict(torch.load(modelpath))
        network.eval()
        for idx,(data_x,data_y,data_index) in enumerate(test_dataloader,0):
            data_x=data_x.to(torch.float32).to(device)
            data_fea=data_x[:,0,:]*data_x[:,1,:]
            output=network(data_fea)
            data_embeds.extend(output.cpu().detach().numpy().tolist())
            data_labels.extend(data_y.tolist())
            data_ids.extend(rawdataset.get_ids(data_index))

def calEntropy(data:list,base=2):
    if not data:
        return 0.0
    
    counts = Counter(data)
    total = len(data)

   
    entropy_val = 0.0
    for count in counts.values():
        p = count / total
        entropy_val -= p * (math.log(p) / math.log(base))
    return entropy_val

def classify(winsize=4,tss_entro_t=1,serdatalistmax=2,clientiplent=3,flowidlenmin=2,lenseridsetmin=1):
    
    clusterdata="./clusters.json"
    with open(clusterdata) as f:
        data=json.load(f)
    
    benignSer=set()
    for serverid,serverdatalist in data.items():
        clientips=set()
        flowids=set()
        for id,flowid,flowlabel,flowts in serverdatalist:
            srcip,dstip,srcport,dstport,proto=id
            if srcip in serverid:
                clientip=dstip
            else:
                clientip=srcip
            clientips.add(clientip)
            flowids.add(flowid)
        if 1 in [item[2] for item in serverdatalist]:
            print(serverid)
            print(len(clientips))
            print(set(clientips))
            print(set(flowids))
            print(len(serverdatalist))
        if len(clientips)>clientiplent and len(flowids)>=flowidlenmin:
            benignSer.add(serverid)
        if len(clientips)<=clientiplent and len(serverdatalist)<=serdatalistmax:
            benignSer.add(serverid)
   
    clientBehavior=dict()
    for serverid,serverdatalist in data.items():
        for id,flowid,flowlabel,flowts in serverdatalist:
            srcip,dstip,srcport,dstport,proto=id
            if srcip in serverid:
                clientip=dstip
            else:
                clientip=srcip
            if clientip not in clientBehavior:
                clientBehavior[clientip]=list()
            clientBehavior[clientip].append([serverid,flowid,flowlabel,flowts])
    
    labels=list()
    pred=list()
    for clientip,clientdata in clientBehavior.items():
        clientdata.sort(key=lambda x: x[3])
        clientdata_tss= np.array([0]+[item[3][0] for item in clientdata])
        clientdata_tss_diff=clientdata_tss[1:]-clientdata_tss[:-1]
        clientdata_tss_all= np.array([item[3] for item in clientdata])
        clientdata_tss_all= np.insert(clientdata_tss_all, 0, 0, axis=1)
        clientdata_tss_all_diff=clientdata_tss_all[:,1:]-clientdata_tss_all[:,:-1]
   
        feas=[item[:3]+[item_ts_dif.tolist(),item_tss_all.tolist()] for item,item_ts_dif,item_ts,item_tss_all in zip(clientdata,clientdata_tss_diff,clientdata_tss,clientdata_tss_all_diff)]
        cliendata_labels=[item[2] for item in clientdata]
        labels.extend(cliendata_labels)
        exitdata=False
        for fea in feas:
            serverid=fea[0]
            if serverid not in benignSer:
                exitdata=True
                break
       
        if not exitdata:
            pred.extend([0]*len(feas))
            continue

        if len(feas)==1:
            pred.extend([0])
            continue
        serverIDS=[item[0] for item in feas]
        serverIDSset=set(serverIDS)

        if len(serverIDSset)==len(serverIDS):
            pred.extend([0]*len(feas))
            continue

        if len(serverIDSset)>lenseridsetmin:
            pred.extend([0]*len(feas))
            continue

        
        if len(feas)<winsize:
            pred.extend([0]*len(feas))
            continue
        tmp_pred=list()
    
        for win_index in range(0,len(feas),winsize):
          
            win_index_max=min(win_index+winsize,len(feas))
            win_fea=feas[win_index:win_index_max]
            win_fea_ids=[item[0] for item in win_fea]
            win_fea_ids_entropy=calEntropy(win_fea_ids)
            if win_fea_ids_entropy>1:
                tmp_pred.extend([0]*(win_index_max-win_index))
                continue
            win_fea_tss=[item[3] for item in win_fea]
            win_fea_tss_entropy=calEntropy(win_fea_tss)
            if win_fea_tss_entropy>tss_entro_t:
                tmp_pred.extend([0]*(win_index_max-win_index))
                continue
            for item in win_fea:
                print(item)
            tmp_pred.extend([1]*(win_index_max-win_index))
        pred.extend(tmp_pred)

    acc=accuracy_score(labels,pred)
    pre=precision_score(labels,pred)
    rec=recall_score(labels,pred)
    f1=f1_score(labels,pred)
    print(acc,pre,rec,f1)

def processFriday16022018():

    # filepath=""
    # modelpath="au.pt"
    # clusterServerdata(filepath,modelpath)

    # clientBehaviorInClusters()
    # clientBehaviorBetwenClusters()

    classify(winsize=10)

def processFriday23022018():

   
    # filepath=""
    # modelpath="au.pt"
    # clusterServerdata(filepath,modelpath)
    # clientBehaviorInClusters()
    # clientBehaviorBetwenClusters()
    classify(winsize=4)

def processThursday15022018():

    filepath=""
    modelpath="au.pt"
    clusterServerdata(filepath,modelpath)

    # clientBehaviorInClusters()
    # clientBehaviorBetwenClusters()

    classify(winsize=3,tss_entro_t=1)


def processTuesday20022018():
    

    classify(winsize=5,tss_entro_t=1,serdatalistmax=2,clientiplent=3,flowidlenmin=5)

def processWednesday14022018():
    

    classify(winsize=10,tss_entro_t=0.5,serdatalistmax=2,clientiplent=3,flowidlenmin=5,lenseridsetmin=2,)

def processWednesday21022018():
    

    classify(winsize=10,tss_entro_t=0.5,serdatalistmax=2,clientiplent=3,flowidlenmin=8,lenseridsetmin=2,)

if __name__ == '__main__':
    
  
    filepath=""
    modelpath="au.pt"
    clusterServerdata(filepath,modelpath)

    # clientBehaviorInClusters()
    # clientBehaviorBetwenClusters()

    # classify(winsize=10,tss_entro_t=0.5,serdatalistmax=2,clientiplent=3,flowidlenmin=8,lenseridsetmin=2,)









