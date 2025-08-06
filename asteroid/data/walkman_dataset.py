import torch
from torch.utils import data
import random
from math import ceil
import librosa
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DATASET = "WalkmanProb"

class WalkmanDataset(data.IterableDataset):
    """Dataset class for on-the-fly mixing"""

    dataset_name = "Walkman"

    class InfiniteIterator():
        def __init__(self, elem_list):
            self.elem_list = np.array(elem_list, dtype=np.bytes_)
            self.inds = np.arange(len(self.elem_list))
            random.shuffle(self.inds)
            self.iterator = iter(self.inds)
        def __next__(self):
            try:
                ind = next(self.iterator)
            except StopIteration:
                random.shuffle(self.inds)
                self.iterator = iter(self.inds)
                ind = next(self.iterator)
            return ind
        def __getitem__(self, ind):
            if self.elem_list[ind].shape:
                return self.elem_list[ind, 0].astype(str), self.elem_list[ind, 1].astype(float)
            else:
                return self.elem_list[ind].astype(str)

    def __init__(self, train_segs, sample_rate=16000, segment=6.0, n_src=2, epoch_size=20000, test=False):
        super(WalkmanProbDataset, self).__init__()
        if test:
            random.seed(1979)
            epoch_size = 3000
        self.fs = sample_rate
        self.segment = segment
        self.n_src = n_src
        self.epoch_size = epoch_size

        self.gpu_ind = None
        self.num_gpus = None

        wav_master_list = []
        with open(train_segs, 'r') as F:
            for line in F:
                t_len = float(line.rstrip().split()[2]) - float(line.split()[1])
                for i in range(int(t_len // self.segment)):
                    wav_master_list.append((line.split()[0], float(line.split()[1]) + i*self.segment))
        self.wav_iterator = self.InfiniteIterator(wav_master_list)
        print(f"walkman loaded {len(wav_master_list)} wav files")

    def __len__(self):
        if self.gpu_ind is not None:
            return self.epoch_size // self.num_gpus
        else:
            return self.epoch_size

    def update_multigpu(self, gpu_ind, num_gpus):
        self.gpu_ind = gpu_ind
        self.num_gpus = num_gpus

    def __iter__(self):
        epoch = np.empty((self.epoch_size, self.n_src), dtype=int)
        for i in range(self.epoch_size):
            for n in range(self.n_src):
                epoch[i, n] = next(self.wav_iterator)

        if self.gpu_ind is not None:
            samples_per_gpu = ceil(len(epoch) / self.num_gpus)
            start_ind = self.gpu_ind * samples_per_gpu
            end_ind = min(start_ind + samples_per_gpu, len(epoch))
            epoch = epoch[start_ind:end_ind]

        worker_info = data.get_worker_info()
        if worker_info is None:  # single-worker
            self.sample_iter = iter(epoch)
        else:  # multi-worker
            samples_per_worker = ceil(len(epoch) / worker_info.num_workers)
            start_ind = worker_info.id * samples_per_worker
            end_ind = min(start_ind + samples_per_worker, len(epoch))
            self.sample_iter = iter(epoch[start_ind:end_ind])

        return self

    def __next__(self):
        sample = next(self.sample_iter)
        source_arrays = []
        for i in range(self.n_src):
            source_path, offset = self.wav_iterator[sample[i]]
            s, _ = librosa.load(source_path, sr=self.fs, offset=offset, duration=self.segment, dtype=np.float32)
            source_arrays.append(s)
        sources = torch.from_numpy(np.vstack(source_arrays))
        x = torch.sum(sources, 0)
        return x, sources

#    def __getitem__(self, idx):
#        # TODO old
#        # Note: this is ONLY for test set
#        source_arrays = []
#        for src_idx in [idx*self.n_src + i for i in range(self.n_src)]
#            source_path, offset = self.voxceleb_master_list[self.voxceleb_inds[src_idx]]
#            s, _ = librosa.load(source_path, sr=self.fs, offset=offset, duration=self.segment, dtype=np.float32)
#            source_arrays.append(s)
#        sources = torch.from_numpy(np.vstack(source_arrays))
#        x = torch.sum(sources, 0)
#        return x, sources
#
#    def mix_info(self, idx):
#        # TODO old
#        return (self.wav_master_list[self.wav_inds[idx*self.n_src+i]] for i in range(self.n_src))

    def get_infos(self):
        infos = dict()
        infos["dataset"] = self.dataset_name
        infos["task"] = "Walkman"
        infos["licenses"] = []
        return infos
