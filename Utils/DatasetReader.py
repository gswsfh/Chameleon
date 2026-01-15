from pathlib import Path
import time
import tqdm
import logging
import os
import socket
import struct
from collections import defaultdict, namedtuple
import pyarrow.parquet as pq
import dpkt
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import pyarrow as pa
import pandas as pd
import numpy as np
import io
from multiprocessing import Pool
from functools import partial


class HyperVisionDatasetReader:
    def __init__(self, datasetdir):
        self.datasetdir = datasetdir

    def read(self, filename):
        time1 = time.time()
        labeldata = Path(os.path.join(
            self.datasetdir, filename+".label")).read_text().strip()
        data_clean, label_clean = list(), list()
        with open(os.path.join(self.datasetdir, filename+".data"), "r", encoding="utf8") as f:
            for l_num, line in enumerate(f.readlines()):
                items = line.strip().split(" ")
                if len(items) > 3:
                    data_clean.append([int(i) for i in items])
                    label_clean.append(int(labeldata[l_num]))
        time2 = time.time()
        return data_clean, label_clean

    def readAll(self):
        filenames = set([item.split(".")[0]
                        for item in os.listdir(self.datasetdir)])
        total_data = list()
        total_label = list()
        for filename in tqdm(filenames):
            data, label = self.read(filename)
            total_data.append(data)
            total_label.append(label)
        return total_data, total_label


def _process_cicids2018_day_wrapper(args):
    datasetdir, day, reprocess = args
    day_path = os.path.join(datasetdir, day)
    output_path = os.path.join(day_path, "allFea.parquet")
    logging.info(f"Processing day: {day}")

    reader = CICDS2018DatasetReader(datasetdir)
    reader._process_day_pcap_to_parquet(day_path, output_path, reprocess)


class CICDS2018DatasetReader:
    SCHEMA = pa.schema([
        ('srcip', pa.string()),
        ('dstip', pa.string()),
        ('srcport', pa.uint16()),
        ('dstport', pa.uint16()),
        ('proto', pa.uint8()),
        ('pktsize', pa.list_(pa.uint32())),
        ('pktts', pa.list_(pa.float64())),
        ('pktdirs', pa.list_(pa.int8())),
    ])
    FlowState = namedtuple(
        'FlowState', 'ts_list size_list dir_list last_ts tcp_done fin_seen rst_seen')
    labelMaps = {
        "Friday-02-03-2018": [
            {
                "att": ["18.219.211.138"],
                "vic": ["18.217.218.111", "172.31.69.23",
                        "18.222.10.237", "172.31.69.17",
                        "18.222.86.193", "172.31.69.14",
                        "18.222.62.221", "172.31.69.12",
                        "13.59.9.106", "172.31.69.10",
                        "18.222.102.2", "172.31.69.8",
                        "18.219.212.0", "172.31.69.6",
                        "18.216.105.13", "172.31.69.26",
                        "18.219.163.126", "172.31.69.29",
                        "18.216.164.12", "172.31.69.30"],
                "start": "10:11",
                "end": "11:34",
                "label": "Bot"
            },
            {
                "att": ["18.219.211.138"],
                "vic": ["18.217.218.111", "172.31.69.23",
                        "18.222.10.237", "172.31.69.17",
                        "18.222.86.193", "172.31.69.14",
                        "18.222.62.221", "172.31.69.12",
                        "13.59.9.106", "172.31.69.10",
                        "18.222.102.2", "172.31.69.8",
                        "18.219.212.0", "172.31.69.6",
                        "18.216.105.13", "172.31.69.26",
                        "18.219.163.126", "172.31.69.29",
                        "18.216.164.12", "172.31.69.30"],
                "start": "14:24",
                "end": "15:55",
                "label": "Bot"
            }
        ],
        "Friday-23-02-2018": [
            {
                "att": ["18.218.115.60"],
                "vic": ["18.218.83.150", "172.31.69.28"],
                "start": "10:03",
                "end": "11:03",
                "label": "Brute Force -Web"
            },
            {
                "att": ["18.218.115.60"],
                "vic": ["18.218.83.150", "172.31.69.28"],
                "start": "13:00",
                "end": "14:10",
                "label": "Brute Force -XSS"
            },
            {
                "att": ["18.218.115.60"],
                "vic": ["18.218.83.150", "172.31.69.28"],
                "start": "15:05",
                "end": "15:18",
                "label": "SQL Injection"
            },
        ],
        "Thursday-15-02-2018": [
            {
                "att": ["172.31.70.46", "18.219.211.138"],
                "vic": ["18.217.21.148", "172.31.69.25"],
                "start": "9:26",
                "end": "10:09",
                "label": "DoS-GoldenEye"
            },
            {
                "att": ["172.31.70.8", "18.217.165.70"],
                "vic": ["18.217.21.148", "172.31.69.25"],
                "start": "10:59",
                "end": "11:40",
                "label": "DoS-Slowloris"
            },
        ],
        "Tuesday-20-02-2018": [
            {
                "att": ["18.218.115.60",
                        "18.219.9.1",
                        "18.219.32.43",
                        "18.218.55.126",
                        "52.14.136.135",
                        "18.219.5.43",
                        "18.216.200.189",
                        "18.218.229.235",
                        "18.218.11.51",
                        "18.216.24.42"],
                "vic": ["18.217.21.148", "172.31.69.25"],
                "start": "10:12",
                "end": "11:17",
                "label": "DDoS attacks-LOIC-HTTP"
            },
            {
                "att": ["18.218.115.60",
                        "18.219.9.1",
                        "18.219.32.43",
                        "18.218.55.126",
                        "52.14.136.135",
                        "18.219.5.43",
                        "18.216.200.189",
                        "18.218.229.235",
                        "18.218.11.51",
                        "18.216.24.42"],
                "vic": ["18.217.21.148", "172.31.69.25"],
                "start": "13:13",
                "end": "13:32",
                "label": "DDoS-LOIC-UDP"
            }
        ],
        "Wednesday-21-02-2018": [
            {
                "att": ["18.218.115.60",
                        "18.219.9.1",
                        "18.219.32.43",
                        "18.218.55.126",
                        "52.14.136.135",
                        "18.219.5.43",
                        "18.216.200.189",
                        "18.218.229.235",
                        "18.218.11.51",
                        "18.216.24.42"],
                "vic": ["18.218.83.150", "172.31.69.28"],
                "start": "10:09",
                "end": "10:43",
                "label": "DDOS-LOIC-UDP"
            },
            {
                "att": ["18.218.115.60",
                        "18.219.9.1",
                        "18.219.32.43",
                        "18.218.55.126",
                        "52.14.136.135",
                        "18.219.5.43",
                        "18.216.200.189",
                        "18.218.229.235",
                        "18.218.11.51",
                        "18.216.24.42"],
                "vic": ["18.218.83.150", "172.31.69.28"],
                "start": "14:05",
                "end": "15:05",
                "label": "DDOS-HOIC"
            }
        ],
        "Friday-16-02-2018": [
            {
                "att": ["172.31.70.23", "13.59.126.31"],
                "vic": ["18.217.21.148", "172.31.69.25"],
                "start": "10:12",
                "end": "11:08",
                "label": "DoS-SlowHTTPTest"
            },
            {
                "att": ["172.31.70.16", "18.219.193.20"],
                "vic": ["18.217.21.148", "172.31.69.25"],
                "start": "13:45",
                "end": "14:19",
                "label": "DoS-Hulk"
            },
        ],
        "Thursday-01-03-2018": [
            {
                "att": ["13.58.225.34"],
                "vic": ["18.216.254.154", "172.31.69.13"],
                "start": "9:57",
                "end": "10:55",
                "label": "Infiltration"
            },
            {
                "att": ["13.58.225.34"],
                "vic": ["18.216.254.154", "172.31.69.13"],
                "start": "14:00",
                "end": "15:37",
                "label": "Infiltration"
            }
        ],
        "Thursday-22-02-2018": [
            {
                "att": ["18.218.115.60"],
                "vic": ["18.218.83.150", "172.31.69.28"],
                "start": "10:17",
                "end": "11:24",
                "label": "Brute Force -Web"
            },
            {
                "att": ["18.218.115.60"],
                "vic": ["18.218.83.150", "172.31.69.28"],
                "start": "13:50",
                "end": "14:29",
                "label": "Brute Force -XSS"
            },
            {
                "att": ["18.218.115.60"],
                "vic": ["18.218.83.150", "172.31.69.28"],
                "start": "16:15",
                "end": "16:29",
                "label": "SQL Injection"
            },
        ],
        "Wednesday-14-02-2018": [
            {
                "att": ["172.31.70.4", "18.221.219.4"],
                "vic": ["172.31.69.25", "18.217.21.148"],
                "start": "10:32",
                "end": "12:09",
                "label": "FTP-BruteForce"
            },
            {
                "att": ["172.31.70.6", "13.58.98.64"],
                "vic": ["172.31.69.25", "18.217.21.148"],
                "start": "14:01",
                "end": "15:31",
                "label": "SSH-Bruteforce"
            },
        ],
        "Wednesday-28-02-2018": [
            {
                "att": ["13.58.225.34"],
                "vic": ["18.221.148.137", "172.31.69.24"],
                "start": "10:50",
                "end": "12:05",
                "label": "Infiltration"
            },
            {
                "att": ["13.58.225.34"],
                "vic": ["18.221.148.137", "172.31.69.24"],
                "start": "13:42",
                "end": "14:40",
                "label": "Infiltration"
            },
        ]
    }
    MAX_PKTS_PER_FLOW = 10000
    dtype_map = {
        'Dst Port': np.int64,
        'Protocol': np.int64,
        'Timestamp': np.object_,
        'Flow Duration': np.int64,
        'Tot Fwd Pkts': np.int64,
        'Tot Bwd Pkts': np.int64,
        'TotLen Fwd Pkts': np.int64,
        'TotLen Bwd Pkts': np.float64,
        'Fwd Pkt Len Max': np.int64,
        'Fwd Pkt Len Min': np.int64,
        'Fwd Pkt Len Mean': np.float64,
        'Fwd Pkt Len Std': np.float64,
        'Bwd Pkt Len Max': np.int64,
        'Bwd Pkt Len Min': np.int64,
        'Bwd Pkt Len Mean': np.float64,
        'Bwd Pkt Len Std': np.float64,
        'Flow Byts/s': np.float64,
        'Flow Pkts/s': np.float64,
        'Flow IAT Mean': np.float64,
        'Flow IAT Std': np.float64,
        'Flow IAT Max': np.float64,
        'Flow IAT Min': np.float64,
        'Fwd IAT Tot': np.float64,
        'Fwd IAT Mean': np.float64,
        'Fwd IAT Std': np.float64,
        'Fwd IAT Max': np.float64,
        'Fwd IAT Min': np.float64,
        'Bwd IAT Tot': np.float64,
        'Bwd IAT Mean': np.float64,
        'Bwd IAT Std': np.float64,
        'Bwd IAT Max': np.float64,
        'Bwd IAT Min': np.float64,
        'Fwd PSH Flags': np.int64,
        'Bwd PSH Flags': np.int64,
        'Fwd URG Flags': np.int64,
        'Bwd URG Flags': np.int64,
        'Fwd Header Len': np.int64,
        'Bwd Header Len': np.int64,
        'Fwd Pkts/s': np.float64,
        'Bwd Pkts/s': np.float64,
        'Pkt Len Min': np.int64,
        'Pkt Len Max': np.int64,
        'Pkt Len Mean': np.float64,
        'Pkt Len Std': np.float64,
        'Pkt Len Var': np.float64,
        'FIN Flag Cnt': np.int64,
        'SYN Flag Cnt': np.int64,
        'RST Flag Cnt': np.int64,
        'PSH Flag Cnt': np.int64,
        'ACK Flag Cnt': np.int64,
        'URG Flag Cnt': np.int64,
        'CWE Flag Count': np.int64,
        'ECE Flag Cnt': np.int64,
        'Down/Up Ratio': np.int64,
        'Pkt Size Avg': np.float64,
        'Fwd Seg Size Avg': np.float64,
        'Bwd Seg Size Avg': np.float64,
        'Fwd Byts/b Avg': np.int64,
        'Fwd Pkts/b Avg': np.int64,
        'Fwd Blk Rate Avg': np.int64,
        'Bwd Byts/b Avg': np.int64,
        'Bwd Pkts/b Avg': np.int64,
        'Bwd Blk Rate Avg': np.int64,
        'Subflow Fwd Pkts': np.int64,
        'Subflow Fwd Byts': np.int64,
        'Subflow Bwd Pkts': np.int64,
        'Subflow Bwd Byts': np.int64,
        'Init Fwd Win Byts': np.int64,
        'Init Bwd Win Byts': np.int64,
        'Fwd Act Data Pkts': np.int64,
        'Fwd Seg Size Min': np.int64,
        'Active Mean': np.float64,
        'Active Std': np.float64,
        'Active Max': np.float64,
        'Active Min': np.float64,
        'Idle Mean': np.float64,
        'Idle Std': np.float64,
        'Idle Max': np.float64,
        'Idle Min': np.float64,
        'Label': np.object_
    }

    dtype_map_cc = {
        'Dst Port': np.int64,
        'Protocol': np.int64,
        'Timestamp': np.object_,
        'Flow Duration': np.int64,
        'Total Fwd Packet': np.int64,
        'Total Bwd packets': np.int64,
        'Total Length of Fwd Packet': np.int64,
        'Total Length of Bwd Packet': np.float64,
        'Fwd Packet Length Max': np.int64,
        'Fwd Packet Length Min': np.int64,
        'Fwd Packet Length Mean': np.float64,
        'Fwd Packet Length Std': np.float64,
        'Bwd Packet Length Max': np.int64,
        'Bwd Packet Length Min': np.int64,
        'Bwd Packet Length Mean': np.float64,
        'Bwd Packet Length Std': np.float64,
        'Flow Bytes/s': np.float64,
        'Flow Packets/s': np.float64,
        'Flow IAT Mean': np.float64,
        'Flow IAT Std': np.float64,
        'Flow IAT Max': np.float64,
        'Flow IAT Min': np.float64,
        'Fwd IAT Total': np.float64,
        'Fwd IAT Mean': np.float64,
        'Fwd IAT Std': np.float64,
        'Fwd IAT Max': np.float64,
        'Fwd IAT Min': np.float64,
        'Bwd IAT Total': np.float64,
        'Bwd IAT Mean': np.float64,
        'Bwd IAT Std': np.float64,
        'Bwd IAT Max': np.float64,
        'Bwd IAT Min': np.float64,
        'Fwd PSH Flags': np.int64,
        'Bwd PSH Flags': np.int64,
        'Fwd URG Flags': np.int64,
        'Bwd URG Flags': np.int64,
        'Fwd Header Length': np.int64,
        'Bwd Header Length': np.int64,
        'Fwd Packets/s': np.float64,
        'Bwd Packets/s': np.float64,
        'Packet Length Min': np.int64,
        'Packet Length Max': np.int64,
        'Packet Length Mean': np.float64,
        'Packet Length Std': np.float64,
        'Packet Length Variance': np.float64,
        'FIN Flag Count': np.int64,
        'SYN Flag Count': np.int64,
        'RST Flag Count': np.int64,
        'PSH Flag Count': np.int64,
        'ACK Flag Count': np.int64,
        'URG Flag Count': np.int64,
        'CWR Flag Count': np.int64,
        'ECE Flag Count': np.int64,
        'Down/Up Ratio': np.int64,
        'Average Packet Size': np.float64,
        'Fwd Segment Size Avg': np.float64,
        'Bwd Segment Size Avg': np.float64,
        'Fwd Bytes/Bulk Avg': np.int64,
        'Fwd Packet/Bulk Avg': np.int64,
        'Fwd Bulk Rate Avg': np.int64,
        'Bwd Bytes/Bulk Avg': np.int64,
        'Bwd Packet/Bulk Avg': np.int64,
        'Bwd Bulk Rate Avg': np.int64,
        'Subflow Fwd Packets': np.int64,
        'Subflow Fwd Bytes': np.int64,
        'Subflow Bwd Packets': np.int64,
        'Subflow Bwd Bytes': np.int64,
        'FWD Init Win Bytes': np.int64,
        'Bwd Init Win Bytes': np.int64,
        'Fwd Act Data Pkts': np.int64,
        'Fwd Seg Size Min': np.int64,
        'Active Mean': np.float64,
        'Active Std': np.float64,
        'Active Max': np.float64,
        'Active Min': np.float64,
        'Idle Mean': np.float64,
        'Idle Std': np.float64,
        'Idle Max': np.float64,
        'Idle Min': np.float64,
        'Label': np.object_
    }

    labelnames = ['Dst Port',
                  'Protocol',
                  'Flow Duration',
                  'Tot Fwd Pkts',
                  'Tot Bwd Pkts',
                  'TotLen Fwd Pkts',
                  'TotLen Bwd Pkts',
                  'Fwd Pkt Len Max',
                  'Fwd Pkt Len Min',
                  'Fwd Pkt Len Mean',
                  'Fwd Pkt Len Std',
                  'Bwd Pkt Len Max',
                  'Bwd Pkt Len Min',
                  'Bwd Pkt Len Mean',
                  'Bwd Pkt Len Std',
                  'Flow Byts/s',
                  'Flow Pkts/s',
                  'Flow IAT Mean',
                  'Flow IAT Std',
                  'Flow IAT Max',
                  'Flow IAT Min',
                  'Fwd IAT Tot',
                  'Fwd IAT Mean',
                  'Fwd IAT Std',
                  'Fwd IAT Max',
                  'Fwd IAT Min',
                  'Bwd IAT Tot',
                  'Bwd IAT Mean',
                  'Bwd IAT Std',
                  'Bwd IAT Max',
                  'Bwd IAT Min',
                  'Fwd PSH Flags',
                  'Bwd PSH Flags',
                  'Fwd URG Flags',
                  'Bwd URG Flags',
                  'Fwd Header Len',
                  'Bwd Header Len',
                  'Fwd Pkts/s',
                  'Bwd Pkts/s',
                  'Pkt Len Min',
                  'Pkt Len Max',
                  'Pkt Len Mean',
                  'Pkt Len Std',
                  'Pkt Len Var',
                  'FIN Flag Cnt',
                  'SYN Flag Cnt',
                  'RST Flag Cnt',
                  'PSH Flag Cnt',
                  'ACK Flag Cnt',
                  'URG Flag Cnt',
                  'CWE Flag Count',
                  'ECE Flag Cnt',
                  'Down/Up Ratio',
                  'Pkt Size Avg',
                  'Fwd Seg Size Avg',
                  'Bwd Seg Size Avg',
                  'Fwd Byts/b Avg',
                  'Fwd Pkts/b Avg',
                  'Fwd Blk Rate Avg',
                  'Bwd Byts/b Avg',
                  'Bwd Pkts/b Avg',
                  'Bwd Blk Rate Avg',
                  'Subflow Fwd Pkts',
                  'Subflow Fwd Byts',
                  'Subflow Bwd Pkts',
                  'Subflow Bwd Byts',
                  'Init Fwd Win Byts',
                  'Init Bwd Win Byts',
                  'Fwd Act Data Pkts',
                  'Fwd Seg Size Min',
                  'Active Mean',
                  'Active Std',
                  'Active Max',
                  'Active Min',
                  'Idle Mean',
                  'Idle Std',
                  'Idle Max',
                  'Idle Min',
                  'Label'
                  ]

    def __init__(self, datasetpath):
        self.datasetpath = datasetpath

    def _flush_batch_to_writer(self, batch_cache, writer):
        if not batch_cache:
            return
        srcip, dstip, srcport, dstport, proto = [], [], [], [], []
        pktsize, pktts, pktdirs = [], [], []
        for key, ts_list, size_list, dir_list in batch_cache:
            s_ip_int, d_ip_int, s_port, d_port, p = key
            srcip.append(socket.inet_ntoa(struct.pack("!I", s_ip_int)))
            dstip.append(socket.inet_ntoa(struct.pack("!I", d_ip_int)))
            srcport.append(s_port)
            dstport.append(d_port)
            proto.append(p)
            pktsize.append(size_list)
            pktts.append(ts_list)
            pktdirs.append(dir_list)
        batch = pa.RecordBatch.from_arrays([
            pa.array(srcip, type=pa.string()),
            pa.array(dstip, type=pa.string()),
            pa.array(srcport, type=pa.uint16()),
            pa.array(dstport, type=pa.uint16()),
            pa.array(proto, type=pa.uint8()),
            pa.array(pktsize, type=pa.list_(pa.uint32())),
            pa.array(pktts, type=pa.list_(pa.float64())),
            pa.array(pktdirs, type=pa.list_(pa.int8())),
        ], schema=self.SCHEMA)
        writer.write_batch(batch)
        batch_cache.clear()

    def _process_pcap_to_parquet_streaming(self, filepath, writer, tcp_timeout=300, udp_timeout=120):
        TH_FIN = dpkt.tcp.TH_FIN
        TH_ACK = dpkt.tcp.TH_ACK
        TH_RST = dpkt.tcp.TH_RST
        ts_list_dict = {}
        size_list_dict = {}
        dir_list_dict = {}
        last_ts_dict = {}
        flags_dict = {}  # [fin_seen, rst_seen, tcp_done]

        batch_cache = []
        BATCH_SIZE = 2000
        MAX_PKTS = self.MAX_PKTS_PER_FLOW

        def flush_flow(key):
            ts_list = ts_list_dict.pop(key, None)
            if ts_list is None:
                return
            size_list = size_list_dict.pop(key)
            dir_list = dir_list_dict.pop(key)
            last_ts_dict.pop(key)
            flags_dict.pop(key)
            batch_cache.append((key, ts_list, size_list, dir_list))
            if len(batch_cache) >= BATCH_SIZE:
                self._flush_batch_to_writer(batch_cache, writer)
        try:
            with open(filepath, 'rb') as f:
                magic = f.read(4)
                f.seek(0)
                if magic in (b'\xd4\xc3\xb2\xa1', b'\xa1\xb2\xc3\xd4'):
                    pcap_reader = dpkt.pcap.Reader(f)
                elif magic == b'\x0a\x0d\x0d\x0a':
                    pcap_reader = dpkt.pcapng.Reader(f)
                else:

                    return

                for ts, buf in pcap_reader:
                    try:
                        eth = dpkt.ethernet.Ethernet(buf)
                        if not isinstance(eth.data, dpkt.ip.IP):
                            continue
                        ip = eth.data
                        if not isinstance(ip, dpkt.ip.IP):
                            continue
                        proto = ip.p
                        ip_data = ip.data

                        if proto == 6 and isinstance(ip.data, dpkt.tcp.TCP):
                            src_port, dst_port, flags = ip_data.sport, ip_data.dport, ip_data.flags
                        elif proto == 17 and isinstance(ip.data, dpkt.udp.UDP):
                            src_port, dst_port, flags = ip_data.sport, ip_data.dport, 0
                        else:
                            continue
                        src_ip = struct.unpack("!I", ip.src)[0]
                        dst_ip = struct.unpack("!I", ip.dst)[0]
                        if (src_ip, src_port) <= (dst_ip, dst_port):
                            key = (src_ip, dst_ip, src_port, dst_port, proto)
                            direction = 1
                        else:
                            key = (dst_ip, src_ip, dst_port, src_port, proto)
                            direction = -1

                        last_ts = last_ts_dict.get(key)
                        if last_ts is not None:
                            idle = ts - last_ts
                            timeout = tcp_timeout if proto == 6 else udp_timeout
                            if idle > timeout:
                                flush_flow(key)
                                last_ts = None

                        if key not in last_ts_dict:
                            ts_list_dict[key] = []
                            size_list_dict[key] = []
                            dir_list_dict[key] = []
                            last_ts_dict[key] = ts
                            flags_dict[key] = [False, False, False]

                        ts_list_dict[key].append(ts)
                        size_list_dict[key].append(len(buf))
                        dir_list_dict[key].append(direction)
                        last_ts_dict[key] = ts

                        if proto == 6:
                            flow_flags = flags_dict[key]
                            if flags & TH_RST:
                                flow_flags[1] = True
                                flow_flags[2] = True
                            elif flags & TH_FIN:
                                flow_flags[0] = True
                            elif flow_flags[0] and (flags & TH_ACK):
                                flow_flags[2] = True

                            if flow_flags[2] or len(ts_list_dict[key]) >= MAX_PKTS:
                                flush_flow(key)

                    except Exception as e:
                        continue
                # flush remaining
                for key in list(ts_list_dict.keys()):
                    flush_flow(key)
                if batch_cache:
                    self._flush_batch_to_writer(batch_cache, writer)
        except Exception as e:
            logging.warning(f"{e}")

    def _write_single_flow_to_writer(self, writer, key, ts_list, size_list, dir_list):
        s_ip_int, d_ip_int, s_port, d_port, p = key
        s_ip = socket.inet_ntoa(struct.pack("!I", s_ip_int))
        d_ip = socket.inet_ntoa(struct.pack("!I", d_ip_int))
        batch = pa.RecordBatch.from_arrays([
            pa.array([s_ip]),
            pa.array([d_ip]),
            pa.array([s_port], type=pa.uint16()),
            pa.array([d_port], type=pa.uint16()),
            pa.array([p], type=pa.uint8()),
            pa.array([size_list], type=pa.list_(pa.uint32())),
            pa.array([ts_list], type=pa.list_(pa.float64())),
            pa.array([dir_list], type=pa.list_(pa.int8())),
        ], schema=self.SCHEMA)
        writer.write_batch(batch)

    def _process_day_pcap_to_parquet(self, day_dir, output_parquet_path, reprocess=False, show_progress=False):
        if os.path.exists(output_parquet_path) and not reprocess:
            return

        pcap_subdir = os.path.join(day_dir, "pcap")
        if not os.path.isdir(pcap_subdir):
            logging.warning(f"No pcap subdir: {day_dir}")
            return

        writer = None
        try:
            writer = pq.ParquetWriter(
                output_parquet_path, self.SCHEMA, compression='snappy')
            pcap_files = [f for f in os.listdir(pcap_subdir)]
            if show_progress:
                from tqdm import tqdm
                pcap_files = tqdm(
                    pcap_files, desc=f"Processing {os.path.basename(day_dir)}")
            for pcap_file in pcap_files:
                pcap_path = os.path.join(pcap_subdir, pcap_file)
                self._process_pcap_to_parquet_streaming(pcap_path, writer)
            logging.info(f"Saved: {output_parquet_path}")
        finally:
            if writer:
                writer.close()

    def groupDayFlowSingle(self, pcapdir, reprocess=False):
        days = [d for d in os.listdir(
            pcapdir) if os.path.isdir(os.path.join(pcapdir, d))]
        for day in days:
            day_path = os.path.join(pcapdir, day)
            output_path = os.path.join(day_path, "allFea.parquet")
            logging.info(f"Processing day: {day}")
            self._process_day_pcap_to_parquet(
                day_path, output_path, reprocess, show_progress=True)

    def _worker_process_day(self, datasetdir, day, reprocess):
        day_path = os.path.join(datasetdir, day)
        output_path = os.path.join(day_path, "allFea.parquet")
        reader = CICDS2018DatasetReader(datasetdir)
        reader._process_day_pcap_to_parquet(
            day_path, output_path, reprocess, show_progress=False)

    def groupDayFlowNulti(self, pcapdir, reprocess=False, n_workers=2):
        days = [d for d in os.listdir(
            pcapdir) if os.path.isdir(os.path.join(pcapdir, d))]
        if not days:
            logging.warning("No day directories found.")
            return

        logging.info(
            f"Starting multiprocessing with {n_workers} workers for {len(days)} days...")
        worker_func = partial(self._worker_process_day,
                              pcapdir, reprocess=reprocess)

        with Pool(processes=n_workers) as pool:
            from tqdm import tqdm
            list(tqdm(
                pool.imap_unordered(worker_func, days),
                total=len(days),
                desc="Processing days in parallel"
            ))
        logging.info("All days processed.")

    def labelFlow(self):
        days = os.listdir(self.datasetpath)
        for day in tqdm(days):
            day_path = os.path.join(self.datasetpath, day)
            allfea_path = os.path.join(day_path, "allFea.parquet")
            flowfea = pd.read_parquet(allfea_path)
            flowfea["label"] = "Benign"
            flowfea['pktts_end'] = flowfea['pktts'].apply(np.max)
            flowfea['pktts_start'] = flowfea['pktts'].apply(np.min)
            flowfea['pktts_start'] = pd.to_datetime(
                flowfea['pktts_start'], unit="s") - pd.Timedelta(hours=4)
            flowfea['pktts_end'] = pd.to_datetime(
                flowfea['pktts_end'], unit="s") - pd.Timedelta(hours=4)
            daymap = CICDS2018DatasetReader.labelMaps[day]
            _, dd, mm, yyyy = day.split('-')
            date_part = f"{yyyy}-{mm}-{dd}"
            for timewin in daymap:
                att_set = set(timewin['att'])
                vic_set = set(timewin['vic'])

                start_time = pd.Timestamp.combine(pd.to_datetime(
                    date_part), pd.to_datetime(timewin['start']).time())
                end_time = pd.Timestamp.combine(pd.to_datetime(
                    date_part), pd.to_datetime(timewin['end']).time())

                forward = flowfea['srcip'].isin(
                    att_set) & flowfea['dstip'].isin(vic_set)
                backward = flowfea['srcip'].isin(
                    vic_set) & flowfea['dstip'].isin(att_set)
                mask = (
                    (forward | backward) &
                    (flowfea['pktts_start'] >= start_time) &
                    (flowfea['pktts_end'] <= end_time)
                )
                flowfea.loc[mask, 'label'] = timewin['label']
            output_path = os.path.join(day_path, "labelFea.parquet")
            flowfea.to_parquet(output_path, index=False)

    def _saveToParquet(self, flowfeatures, savepath):

        alldata = []

        for flowid, flowfea in flowfeatures.items():
            srcip, dstip, srcport, dstport, proto = flowid  #

            pktsize = [item["size"] for item in flowfea]
            pktts = [item["ts"] for item in flowfea]

            row_df = pd.DataFrame({
                "srcip": [srcip],
                "dstip": [dstip],
                "srcport": [srcport],
                "dstport": [dstport],
                "proto": [proto],
                "pktsize": [pktsize],
                "pktts": [pktts]
            })
            alldata.append(row_df)
        final_df = pd.concat(alldata, ignore_index=True)
        final_df.to_parquet(savepath, index=False)

    def _readFromParquet(self, filepath):
        data = pd.read_parquet(filepath)
        return data

    def readMetaDataset(self, file):
        return self._readFromParquet(os.path.join(self.datasetpath, "Original Network Traffic and Log data", file, "labelFea.parquet"))

    def readCsvDataset(self, file, feaDisable=None):
        with open(os.path.join(self.datasetpath, "Processed Traffic Data for ML Algorithms", file), 'r', encoding='utf-8', errors='ignore') as f:
            filtered_lines = [line for i, line in enumerate(
                f) if i == 0 or 'Dst' not in line]
        time00 = time.time()
        datasetRaw = pd.read_csv(
            io.StringIO(''.join(filtered_lines)),
            dtype=self.dtype_map,
            on_bad_lines='skip',
            usecols=self.dtype_map.keys(),
        )
        time0 = time.time()

        le = LabelEncoder()
        datasetRaw["label_NUM"] = le.fit_transform(datasetRaw['Label'])

        datasetRaw = datasetRaw.drop(columns=['Label', 'Timestamp'])

        datasetRaw = datasetRaw.replace([np.inf, -np.inf], -1).fillna(-1)

        datasetRawlist = datasetRaw.values
        x, y = datasetRawlist[:, :-1], datasetRawlist[:, -1]
        time1 = time.time()
        mapping = dict(zip(le.classes_, le.transform(le.classes_)))

        if feaDisable:
            indexs = list()
            for feature in feaDisable:
                feaindex = self.labelnames.index(feature)
                indexs.append(feaindex)
            indexs_remain = list(
                set([i for i in range(len(x[0]))])-set(indexs))
            x = x[:, indexs_remain]

        return x, y

    def readMultiCsvDataset(self, files):

        df_list = list()
        for file in tqdm(files):

            with open(os.path.join(self.datasetpath, "Processed Traffic Data for ML Algorithms", file), 'r', encoding='utf-8', errors='ignore') as f:
                filtered_lines = [line for i, line in enumerate(
                    f) if i == 0 or 'Dst' not in line]

            time00 = time.time()
            datasetRaw = pd.read_csv(
                io.StringIO(''.join(filtered_lines)),
                dtype=self.dtype_map,
                on_bad_lines='skip',
                usecols=self.dtype_map.keys(),
            )
            time0 = time.time()

            df_list.append(datasetRaw)
        datasetRaw = pd.concat(df_list, ignore_index=True)

        le = LabelEncoder()
        datasetRaw["label_NUM"] = le.fit_transform(datasetRaw['Label'])

        datasetRaw = datasetRaw.drop(columns=['Label', 'Timestamp'])

        datasetRaw = datasetRaw.replace([np.inf, -np.inf], -1).fillna(-1)

        datasetRawlist = datasetRaw.values
        x, y = datasetRawlist[:, :-1], datasetRawlist[:, -1]
        time1 = time.time()
        mapping = dict(zip(le.classes_, le.transform(le.classes_)))

        return x, y

    def directReadCCFlowMeterCsv(self, filename, feaDisable=None):
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            filtered_lines = [line for i, line in enumerate(
                f) if i == 0 or 'Dst' not in line]

        time00 = time.time()
        datasetRaw = pd.read_csv(
            io.StringIO(''.join(filtered_lines)),
            dtype=self.dtype_map,
            on_bad_lines='skip',
            usecols=self.dtype_map_cc.keys(),
        )
        time0 = time.time()

        le = LabelEncoder()
        datasetRaw["label_NUM"] = le.fit_transform(datasetRaw['Label'])

        datasetRaw = datasetRaw.drop(columns=['Label', 'Timestamp'])

        datasetRaw = datasetRaw.replace([np.inf, -np.inf], -1).fillna(-1)

        datasetRawlist = datasetRaw.values
        x, y = datasetRawlist[:, :-1], datasetRawlist[:, -1]
        time1 = time.time()
        mapping = dict(zip(le.classes_, le.transform(le.classes_)))

        if feaDisable:
            indexs = list()
            for feature in feaDisable:
                feaindex = self.labelnames.index(feature)
                indexs.append(feaindex)
            indexs_remain = list(
                set([i for i in range(len(x[0]))])-set(indexs))
            x = x[:, indexs_remain]

        return x, y


class GeneralPcapDatasetReader:
    SCHEMA = pa.schema([
        ('srcip', pa.string()),
        ('dstip', pa.string()),
        ('srcport', pa.uint16()),
        ('dstport', pa.uint16()),
        ('proto', pa.uint8()),
        ('pktsize', pa.list_(pa.uint32())),
        ('pktts', pa.list_(pa.float64())),
        ('pktts_start', pa.float64()),
        ('pktts_end', pa.float64()),
        ('pktdirs', pa.list_(pa.int8())),
    ])

    def __init__(self, datasetpath):
        self.datasetpath = datasetpath

    def _extract_and_group_to_flows_streaming(self, filepath, maxpkt_num=None):

        flows = defaultdict(list)
        count = 0

        try:
            with open(filepath, 'rb') as f:

                magic = f.read(4)
                f.seek(0)
                if magic in (b'\xd4\xc3\xb2\xa1', b'\xa1\xb2\xc3\xd4'):
                    pcap_reader = dpkt.pcap.Reader(f)
                elif magic == b'\x0a\x0d\x0d\x0a':
                    pcap_reader = dpkt.pcapng.Reader(f)
                else:
                    logging.error(f"Unknown format: {filepath}")
                    return flows

                for ts, buf in pcap_reader:
                    if maxpkt_num and count >= maxpkt_num:
                        break
                    count += 1

                    try:
                        eth = dpkt.ethernet.Ethernet(buf)
                        if not isinstance(eth.data, dpkt.ip.IP):
                            continue
                        ip = eth.data
                        if not isinstance(ip, dpkt.ip.IP):
                            continue

                        src_ip = socket.inet_ntoa(ip.src)
                        dst_ip = socket.inet_ntoa(ip.dst)
                        proto = ip.p
                        pkt_len = len(buf)

                        src_port = dst_port = None
                        trans_proto = None
                        if isinstance(ip.data, dpkt.tcp.TCP):
                            tcp = ip.data
                            src_port, dst_port = tcp.sport, tcp.dport
                            trans_proto = 'tcp'
                        elif isinstance(ip.data, dpkt.udp.UDP):
                            udp = ip.data
                            src_port, dst_port = udp.sport, udp.dport
                            trans_proto = 'udp'
                        else:
                            continue

                        if (src_ip, src_port) <= (dst_ip, dst_port):
                            flow_key = (src_ip, dst_ip, src_port,
                                        dst_port, proto)
                            direction = 1
                        else:
                            flow_key = (dst_ip, src_ip, dst_port,
                                        src_port, proto)
                            direction = -1

                        flows[flow_key].append((pkt_len, ts, direction))

                    except (dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError, AttributeError, struct.error):
                        continue

        except Exception as e:
            logging.warning(f"Error reading {filepath}: {e}")

        return flows

    def _write_flows_to_parquet(self, flows, writer):

        if not flows:
            return

        srcip, dstip, srcport, dstport, proto = [], [], [], [], []
        pktsize, pktts, pktdirs = [], [], []
        pktts_start, pktts_end = [], []

        for (s_ip, d_ip, s_port, d_port, p), pkts in flows.items():
            srcip.append(s_ip)
            dstip.append(d_ip)
            srcport.append(s_port)
            dstport.append(d_port)
            proto.append(p)
            pktsize.append([pkt[0] for pkt in pkts])
            tss = [pkt[1] for pkt in pkts]
            pktts.append(tss)
            pktdirs.append([pkt[2] for pkt in pkts])
            pktts_start.append(np.min(tss))
            pktts_end.append(np.max(tss))
        batch = pa.RecordBatch.from_arrays([
            pa.array(srcip, type=pa.string()),
            pa.array(dstip, type=pa.string()),
            pa.array(srcport, type=pa.uint16()),
            pa.array(dstport, type=pa.uint16()),
            pa.array(proto, type=pa.uint8()),
            pa.array(pktsize, type=pa.list_(pa.uint32())),
            pa.array(pktts, type=pa.list_(pa.float64())),
            pa.array(pktts_start, type=pa.float64()),
            pa.array(pktts_end, type=pa.float64()),
            pa.array(pktdirs, type=pa.list_(pa.int8())),
        ], schema=self.SCHEMA)
        writer.write_batch(batch)

    def _process_day_pcap_to_parquet(self, filepath, output_parquet_path):

        writer = None
        try:
            writer = pq.ParquetWriter(
                output_parquet_path, self.SCHEMA, compression='snappy')
            flows = self._extract_and_group_to_flows_streaming(filepath)
            if flows:
                self._write_flows_to_parquet(flows, writer)

            logging.info(f"Saved: {output_parquet_path}")
        finally:
            if writer:
                writer.close()

    def groupFlow(self, pcapname):

        filepath = os.path.join(self.datasetpath, pcapname)
        output_path = os.path.join(
            self.datasetpath, pcapname.split(".")[0]+".parquet")
        self._process_day_pcap_to_parquet(filepath, output_path)

    def _saveToParquet(self, flowfeatures, savepath):

        alldata = []

        for flowid, flowfea in flowfeatures.items():
            srcip, dstip, srcport, dstport, proto = flowid

            pktsize = [item["size"] for item in flowfea]
            pktts = [item["ts"] for item in flowfea]

            row_df = pd.DataFrame({
                "srcip": [srcip],
                "dstip": [dstip],
                "srcport": [srcport],
                "dstport": [dstport],
                "proto": [proto],
                "pktsize": [pktsize],
                "pktts": [pktts]
            })
            alldata.append(row_df)
        final_df = pd.concat(alldata, ignore_index=True)
        final_df.to_parquet(savepath, index=False)

    def _readFromParquet(self, filepath):
        data = pd.read_parquet(filepath)
        return data

    def read(self, file):
        return self._readFromParquet(os.path.join(self.datasetpath, file))


if __name__ == '__main__':
    # datasetdir=""
    # reader=GeneralPcapDatasetReader(datasetdir)
    # reader.groupFlow("201811041400.pcap")

    datasetdir = ""
    reader = CICDS2018DatasetReader(datasetdir)
    # reader.groupDayFlowSingle(datasetdir,True)
    # reader.groupDayFlowNulti(datasetdir,True, n_workers=10)
    reader.labelFlow()
