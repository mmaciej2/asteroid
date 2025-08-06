import torch
from torch.utils import data
import random
from math import ceil
import librosa
import numpy as np
from scipy.signal import convolve

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DATASET = "WalkmanProb"

class WalkmanProbDataset(data.IterableDataset):
    """Dataset class for on-the-fly VoxCeleb mixing with noise/reverb augmentation"""

    dataset_name = "WalkmanProb"

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

    def __init__(self, train_segs, noise_segs, rir_list, aug_probs={"noise": 0.4, "reverb": 0.4}, n_spk_probs={1: 0.15, 2: 0.45, 3: 0.4}, denoise=True, derev=False, noise_std=1e-6, sample_rate=16000, segment=6.0, n_src=2, epoch_size=20000, test=False):
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

        self.denoise = denoise
        if derev:
            raise NotImplementedError
        self.noise_std = noise_std

        self.aug_probs = aug_probs
        self.n_spk_probs = n_spk_probs
        assert(sum(n_spk_probs.values()) == 1.0)

        wav_master_list = []
        with open(train_segs, 'r') as F:
            for line in F:
                t_len = float(line.rstrip().split()[2]) - float(line.split()[1])
                for i in range(int(t_len // self.segment)):
                    wav_master_list.append((line.split()[0], float(line.split()[1]) + i*self.segment))
        self.wav_iterator = self.InfiniteIterator(wav_master_list)
        print(f"walkman loaded {len(wav_master_list)} wav files")

        noise_master_list = []
        with open(noise_segs, 'r') as F:
            for line in F:
                t_len = float(line.rstrip().split()[2]) - float(line.split()[1])
                for i in range(int(t_len // self.segment)):
                    noise_master_list.append((line.split()[0], float(line.split()[1]) + i*self.segment))
        self.noise_iterator = self.InfiniteIterator(noise_master_list)

        rir_master_list = []
        with open(rir_list, 'r') as F:
            for line in F:
                rir_master_list.append(line.rstrip())
        self.rir_iterator = self.InfiniteIterator(rir_master_list)

    def __len__(self):
        if self.gpu_ind is not None:
            return self.epoch_size // self.num_gpus
        else:
            return self.epoch_size

    def update_multigpu(self, gpu_ind, num_gpus):
        self.gpu_ind = gpu_ind
        self.num_gpus = num_gpus

    def __iter__(self):
        epoch = -np.ones((self.epoch_size, self.n_src*2+1), dtype=int)
        # Samples are: [wav1, rir1, wav2, rir2, ..., wavN, rirN, noise] with -1 if not using
        for i in range(self.epoch_size):
            n_spk_val = random.uniform(0, 1)
            use_reverb = random.uniform(0, 1) < self.aug_probs["reverb"]
            for n in range(0, 2*self.n_src, 2):
                if n_spk_val > 0:
                    epoch[i, n] = next(self.wav_iterator)
                    if use_reverb:
                        epoch[i, n+1] = next(self.rir_iterator)
                n_spk_val -= self.n_spk_probs[n//2+1]
            if random.uniform(0, 1) < self.aug_probs["noise"]:
                epoch[i, -1] = next(self.noise_iterator)

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
        for i in range(0, self.n_src*2, 2):
            if sample[i] >= 0:
                source_path, offset = self.wav_iterator[sample[i]]
                s, _ = librosa.load(source_path, sr=self.fs, offset=offset, duration=self.segment, dtype=np.float32)
                if sample[i+1] >= 0:
                    rir_path = self.rir_iterator[sample[i+1]]
                    rir, _ = librosa.load(rir_path, sr=self.fs, dtype=np.float32)
                    power = (s ** 2).mean()
                    s = convolve(s, rir, mode="same")
                    power2 = (s ** 2).mean()
                    s = np.sqrt(power / max(power2, 1e-10)) * s
                source_arrays.append(s)
            else:
                s = np.random.normal(scale=self.noise_std, size=len(source_arrays[0])).astype(np.float32)
                source_arrays.append(s)
        sources = torch.from_numpy(np.vstack(source_arrays))
        x = torch.sum(sources, 0)
        if sample[-1] >= 0:
            noise_path, offset = self.noise_iterator[sample[-1]]
            n, _ = librosa.load(noise_path, sr=self.fs, offset=offset, duration=self.segment, dtype=np.float32)
            x = x + torch.from_numpy(n)
            if not self.denoise:
                for i in range(len(sources)):
                    sources[i] = sources[i] + torch.from_numpy(n)
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
        infos["task"] = "WalkmanProb"
        infos["licenses"] = []
        return infos
