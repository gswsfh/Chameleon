import datetime
import torch.nn.functional as F
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, random_split, DataLoader
import os
import logging
import pickle
import awkward as ak
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

class TrafficMetadataDataset(Dataset):
    FLOWMAXLENGTH=32
    PAD_VAL=0
    TST=0.1
    def __init__(self,filepath,cache_dir="./embedcache", random_seed=42,sample=False,sample_num=10*10000):
        super(Dataset, self).__init__()
        np.random.seed(random_seed)
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_path_pre=os.path.join(self.cache_dir,filepath.split("/")[-2])

        taskfilepath=self.cache_path_pre+"-tasks.pkl"
        feafilepath=self.cache_path_pre+"-fea.pkl"
        idsfilepath=self.cache_path_pre+"-ids.pkl"
        if os.path.exists(self.cache_path_pre+"-fea.pkl"):
            with open(taskfilepath, 'rb') as f:
                self.fea = pickle.load(f)
            with open(feafilepath, 'rb') as f:
                self.label = pickle.load(f)
            with open(idsfilepath, 'rb') as f:
                self.ids = pickle.load(f)
            if sample:
                rng = np.random.default_rng(seed=42)  
                sampled_indices = rng.choice(len(self.label), size=sample_num, replace=False)
                self.fea = self.fea[sampled_indices]
                self.label = self.label[sampled_indices]
                self.ids = self.ids[sampled_indices]
        else:
          
            self.fea,self.label,self.ids=self._build_flow_feas(filepath,sample,sample_num)
            # self.ids = self.ids.values
            with open(taskfilepath, 'wb') as f:
                pickle.dump(self.fea,f)
            with open(feafilepath, 'wb') as f:
                pickle.dump(self.label,f)
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
        # X_dir = ak.to_numpy(ak.where(dirs == -1, 0.5, dirs)).astype(np.float32)
        X_dir = dirs.to_numpy().astype(np.float32)
  
        X = np.stack([X_pkt, X_dir],axis=1).astype(np.float32)
        X_task=self._build_task_samples(X)
     
        ids_np = np.array([
            tuple(row.values()) for row in ak.to_list(tbl[["srcip", "dstip", "srcport", "dstport", "proto"]])
        ], dtype=[
            ('srcip', 'U15'),
            ('dstip', 'U15'),
            ('srcport', 'i4'),
            ('dstport', 'i4'),
            ('proto', 'i4')
        ])
        return X_task,X[:,0,:],ids_np

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

    def __getitem__(self, index):
        return self.fea[index],self.label[index],index

    def get_id(self,index):
        return self.ids[index]

    def get_ids(self,indexs):
        return self.ids[indexs]

    def __len__(self) -> int:
        return len(self.label)

class FeatureAttention1D(nn.Module):
  
    def __init__(self, feature_dim, hidden=None):
        super().__init__()
        if hidden is None:
            hidden = max(1, feature_dim // 4)
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, feature_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
 
        if x.size(1) == 1:
            x_squeezed = x.squeeze(1)  # (B, L)
            weights = self.net(x_squeezed)  # (B, L)
            x = x * weights.unsqueeze(1)  # (B, 1, L)
        else:
     
            x_mean = x.mean(dim=1)  # (B, L)
            weights = self.net(x_mean)  # (B, L)
            x = x * weights.unsqueeze(1)
        return x

class ResidualConvAttentionBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, hidden_att=None):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels)
        )
        self.attention = FeatureAttention1D(channels, hidden_att)
        self.relu = nn.ReLU()

    def forward(self, x):
    
        residual = x
        out = self.conv(x)
        out = self.attention(out)  # (B, C, L)
        out = self.relu(out + residual)  
        return out

class CNNAutoencoderWithAttention(nn.Module):
    def __init__(self, feature_size, latent_dim=64, num_blocks=2):
        super().__init__()
        self.feature_size = feature_size

        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        
        )
        self.encoder_res_blocks = nn.Sequential(
            *[ResidualConvAttentionBlock(32, kernel_size=3) for _ in range(num_blocks)]
        )
  
        self.flat_dim = self._get_flat_dim(feature_size)
        self.encoder_fc = nn.Linear(self.flat_dim, latent_dim)

     
        self.decoder_fc = nn.Linear(latent_dim, self.flat_dim)
        self.decoder_upsample = nn.Upsample(size=feature_size, mode='linear', align_corners=True)
        self.decoder_conv = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=3, padding=1)
        )

    def _get_flat_dim(self, feat_size):
        x = torch.zeros(1, 1, feat_size)
        x = self.encoder_conv(x)
        return x.numel()

    def forward(self, x):
        # x: (B, F)
        x = x.unsqueeze(1)  # (B, 1, F)

        # Encoder
        enc = self.encoder_conv(x)  # (B, 32, L1)
        enc = self.encoder_res_blocks(enc)  # (B, 32, L1)
        enc_flat = enc.view(enc.size(0), -1)
        z = self.encoder_fc(enc_flat)  # (B, latent_dim)

        # Decoder
        dec = self.decoder_fc(z)  # (B, flat_dim)
        dec = dec.view(enc.shape)  # (B, 32, L1)
        dec = self.decoder_upsample(dec)  # (B, 32, F)
        dec = self.decoder_conv(dec)  # (B, 1, F)
        x_recon = dec.squeeze(1)  # (B, F)

        return x_recon, z

class Network(nn.Module):
    def __init__(self, feature_size, latent_dim=128):
        super(Network, self).__init__()
        self.autoencoder = CNNAutoencoderWithAttention(feature_size, latent_dim)
    def forward(self, x):
        decodes, z = self.autoencoder(x)
        return decodes, z

class MaskedMSELoss(nn.Module):
    def __init__(self, pad_value=0.0, reduction='mean'):
        super().__init__()
        self.pad_value = pad_value
        self.reduction = reduction

    def forward(self, pred, target):
        # pred, target: (B, F)
        mask = (target != self.pad_value)  # (B, F), bool
        diff = (pred - target) ** 2        # (B, F)

        if self.reduction == 'mean':
        
            loss = (diff * mask).sum() / mask.sum().clamp(min=1)
        elif self.reduction == 'sum':
            loss = (diff * mask).sum()
        else:
            loss = diff * mask  

        return loss

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5, eps=1e-8):

        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, z_i, z_j):
 
        batch_size = z_i.size(0)

    
        z_i = F.normalize(z_i, dim=1, eps=self.eps)
        z_j = F.normalize(z_j, dim=1, eps=self.eps)

       
        z = torch.cat([z_i, z_j], dim=0)  # shape: (2N, D)

        
        sim = torch.mm(z, z.t()) / self.temperature  # shape: (2N, 2N)

        
        sim_i_j = torch.diag(sim, diagonal=batch_size)         
        sim_j_i = torch.diag(sim, diagonal=-batch_size)         

        positive_samples = torch.cat([sim_i_j, sim_j_i], dim=0)  


        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask, float('-inf'))  


        denominator = torch.logsumexp(sim, dim=1)  # shape: (2N,)

        # NT-Xent loss: -log( exp(pos) / sum(exp(all)) ) = -(pos - logsumexp(all))
        loss = -positive_samples + denominator
        return loss.mean()

if __name__ == '__main__':
    learning_rate=0.01
    epochsize=100
    batchsize=1024
    step=1 
    modelpath="embed.pt"
    sampled=True
    samplenum=10*10000
    trainrate=0.7
    now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network=Network(TrafficMetadataDataset.FLOWMAXLENGTH).to(device)
    criterion_con_mse=nn.MSELoss()
    # criterion_au=MaskedMSELoss()
    criterion_con_nextent=NTXentLoss(0.1)
    optimizer=torch.optim.Adam(network.parameters(),lr=learning_rate)

    filepath="filepath"
    rawdataset=TrafficMetadataDataset(filepath,sample=sampled,sample_num=samplenum)
    train_size = int(trainrate * len(rawdataset))
    test_size = len(rawdataset) - train_size
    traindataset,testdataset=random_split(rawdataset,[train_size,test_size])
    train_dataloader = DataLoader(dataset=traindataset, batch_size=batchsize, shuffle=True)
    test_dataloader = DataLoader(dataset=testdataset, batch_size=batchsize, shuffle=False)
    if step==0:
        writer = SummaryWriter(log_dir='./runs')
        network.train()
        for epoch in range(0,epochsize):
            network.train()
            epoch_loss=list()
            for idx,(data_tasks,data_fea,data_index) in enumerate(train_dataloader,0):
                data_tasks=data_tasks.to(torch.float32).to(device)
                data_fea=data_fea.to(torch.float32).to(device)
                data_fea_output,data_fea_z=network(data_fea)
                # loss=criterion_au(data_fea_output,data_fea)
                loss=criterion_con_mse(data_fea_output,data_fea)
                for i in range(4):
                    data_task_i_output,data_task_i_z=network(data_tasks[:,i,:])
                    # loss += criterion_au(data_task_i_output,data_tasks[:,i,:])
                    loss += criterion_con_mse(data_task_i_output,data_tasks[:,i,:])
                    loss += criterion_con_mse(data_fea_output,data_task_i_output)
                    loss += criterion_con_mse(data_fea_z,data_task_i_z)
              
                    loss += criterion_con_nextent(data_task_i_output,data_tasks[:,i,:])
                    loss += criterion_con_nextent(data_fea_output,data_task_i_output)
                    loss += criterion_con_nextent(data_fea_z,data_task_i_z)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())
            writer.add_scalar(now_str+'Training Loss', np.mean(epoch_loss).item(),epoch)
            print(f"epoch={epoch}/{epochsize},loss={np.mean(epoch_loss)}")
            torch.save(network.state_dict(),modelpath)
    elif step==1:
        data_embeds=list()
        data_ids=list()
        network.load_state_dict(torch.load(modelpath))
        network.eval()
        for idx,(data_tasks,data_fea,data_index) in enumerate(test_dataloader,0):
            data_fea=data_fea.to(torch.float32).to(device)
            data_decode,data_z=network(data_fea)
            data_embeds.extend(data_z.cpu().detach().numpy().tolist())
            data_ids.extend(rawdataset.get_ids(data_index))
        print(data_embeds)
        print(data_ids)
