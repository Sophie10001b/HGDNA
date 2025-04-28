import os
import re
import glob
import argparse
import pandas as pd
import numpy as np
import torch
import pysam
import random

from itertools import product
from torch.utils.data import Dataset, DataLoader
from typing import Optional
from copy import deepcopy
from transformers import AutoTokenizer, PreTrainedTokenizer
from lightning.pytorch import LightningDataModule

BASEDICT = {'A': 0, 'C': 1, 'G': 2, 'T': 3, '<s>': 4, '</s>': 5, '<pad>': 6, '<unk>': 7, '<mask>': 8, '|': 9}
CLASSLIST = [f"<class_{i}>" for i in range(1024)]

for _ in CLASSLIST: BASEDICT[_] = len(BASEDICT)

EXTRASET = set(['<s>', '</s>', '<pad>', '<unk>', '<mask>', '|'] + CLASSLIST)

REVERSEDICT = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}

class PackedData():
    """
    seq: (B, L) array
    label: (B * Lmask,) for MLM, (B * L,) for NTP, or (B,) for "classification" task
    maskIdx: (B * Lmask,) for MLM or None for other case
    speciesIdx: (B,) for pretraining or None for other case
    promptIdx: (B,)
    augment: bool
    """
    def __init__(self, seq: np.ndarray, label: np.ndarray, maskIdx: Optional[np.ndarray]=None, speciesIdx: Optional[np.ndarray]=None, task: Optional[str]="", promptIdx: Optional[np.ndarray]=None, augment: Optional[bool]=False):
        self.seq = torch.tensor(seq, dtype=torch.int64)
        self.label = torch.tensor(label, dtype=torch.int64)
        self.maskIdx = torch.tensor(maskIdx, dtype=torch.int64) if maskIdx is not None else None
        self.speciesIdx = torch.tensor(speciesIdx, dtype=torch.int64) if speciesIdx is not None else None
        self.task = task
        self.promptIdx = torch.tensor(promptIdx, dtype=torch.int64) if promptIdx is not None else None
        self.augment = augment
    
    def __len__(self):
        return len(self.seq)
    
    def to(self, device: str):
        self.seq = self.seq.to(device)
        self.label = self.label.to(device)
        if self.maskIdx is not None: self.maskIdx = self.maskIdx.to(device)
        if self.speciesIdx is not None: self.speciesIdx = self.speciesIdx.to(device)
        if self.promptIdx is not None: self.promptIdx = self.promptIdx.to(device)
        return self
    
    def pinMemory(self):
        self.seq = self.seq.pin_memory()
        self.label = self.label.pin_memory()
        if self.maskIdx is not None: self.maskIdx = self.maskIdx.pin_memory()
        if self.speciesIdx is not None: self.speciesIdx = self.speciesIdx.pin_memory()
        if self.promptIdx is not None: self.promptIdx = self.promptIdx.pin_memory()
        return self

def padding(seqList: list[np.ndarray] | np.ndarray, vocab: dict[str:int], maxLen: Optional[int]=-1):
    if maxLen == -1:
        maxLen = max([len(_) for _ in seqList])

    if isinstance(seqList, list):
        for i in range(len(seqList)):
            if len(seqList[i]) < maxLen:
                seqList[i] = np.pad(seqList[i], (0, maxLen - len(seqList[i])), mode="constant", constant_values=(vocab['<pad>'], vocab['<pad>']))
    elif isinstance(seqList, np.ndarray):
        if len(seqList) < maxLen:
            seqList = np.pad(seqList, (0, maxLen - len(seqList)), mode="constant", constant_values=(vocab['<pad>'], vocab['<pad>']))
    return seqList

def tokenizing(src: str, tokenization: str, vocab: dict[str:int], k: int, stride: int, bos: Optional[list[str]]=[], eos: Optional[list[str]]=[], tokenizeModel=None) -> np.ndarray:
    """
    return tokenized array
    """
    if src.startswith('|'): src = src[1:]
    if src.endswith('|'): src = src[:-1]

    if tokenization == "BPE":
        if len(bos) > 0: bos = "".join(bos)
        if len(eos) > 0: eos = "".join(eos)
        src = tokenizeModel.encode(bos + src + eos, add_special_tokens=False)
    else:
        src = src.split('|')
        new_src = []

        for i in range(len(src)):
            _src = src[i]
            if tokenization == "base":
                res = [vocab.get(_, vocab['<unk>']) for _ in _src]
            elif "mer" in tokenization:
                res = []
                pos = 0
                for pos in range(0, len(_src) - k + 1, stride):
                    if len(_src) - pos >= k:
                        kmer = _src[pos:pos+k]
                        res.append(vocab.get(kmer, vocab['<unk>']))
                
                if stride > 1 or len(_src) < k:
                    for pos in range(pos+1, len(_src)):
                        res.append(vocab[_src[pos]])

            new_src.extend(res + [vocab['|']])

        src = new_src[:-1]
        if len(bos) > 0: src = [vocab[_] for _ in bos] + src
        if len(eos) > 0: src = src + [vocab[_] for _ in eos]

    return np.array(src, dtype=np.int64)

def detokenizing(src: np.ndarray, tokenization: str, vocab: dict[str:int], k: int, stride: int, tokenizeModel: PreTrainedTokenizer=None) -> list[str]:
    """
    src: size(N, L)
    """
    rvocab = {v: k for k, v in vocab.items()}
    res = []
    for _src in src:
        _src = _src.tolist()

        if tokenizeModel is not None:
            _src = tokenizeModel.decode(_src, skip_special_tokens=True)
        else:
            for i in range(len(_src)):
                _word = rvocab[_src[i]]
                if _word in EXTRASET: _src[i] = ''
                else:
                    if tokenization == "base" or len(_word) <= stride:
                        _src[i] = _word
                    elif stride == k: _src[i] = _word
                    elif stride == 1:
                        _src[i] = _word if i == 0 else _word[-1]

            _src = "".join(_src)
        
        res.append(_src)
    return res

def reverse_complement(src: str, apply: bool=False) -> str:
    if apply:
        src = list(src)
        src.reverse()
        src = [REVERSEDICT[_] if _ in REVERSEDICT else _ for _ in src]
        src = ''.join(src)
    return src


class BaseDataset(Dataset):
    def __init__(self, dataPath: str, mode: str, args: argparse.Namespace):
        super(BaseDataset, self).__init__()
        self.dataPath = dataPath
        self.mode = mode
        self.trainMode = args.trainMode
        self.minSeqLen = args.minSeqLen
        self.maxSeqLen = args.maxSeqLen
        self.tokenization: str = args.tokenization
        self.pretrainTask = args.pretrainTask
        self.finetuneTask = args.finetuneTask
        self.augment = args.augment
        self.numWorker = max(1, args.numWorker)
        self.loadChunk = int(5e9)

        self.args = args

        self._step = 0
        self.k = -1
        self.stride = -1
        self.tokenizeModel = None

        # [k]mer_[stride]stride
        # 1st token rep k bp, the left token rep stride bp, which means give a [n]-length tokens, the compression rate is near to ([k] + [stride]*[n-1]) / [n] -> [stride]
        if 'mer' in self.tokenization:
            kmer, stride = self.tokenization.split("_")
            self.k = int(kmer[0])
            self.stride = int(stride[0])
            assert self.stride <= self.k
            self.vocabSize = len(BASEDICT) + 4 ** self.k
            
            vocab = product("ACGT", repeat=6)
            mapping = lambda x: sum([BASEDICT[_] * (4 ** i) for i, _ in enumerate(x)])
            
            self.vocab = deepcopy(BASEDICT)
            for v in vocab:
                self.vocab[''.join(v)] = mapping(v) + len(BASEDICT)

        elif self.tokenization == "BPE":
            bpePath = os.path.dirname(os.path.realpath(__file__))
            bpePath = os.path.join(bpePath, "bpe")
            bpeWrapper = AutoTokenizer.from_pretrained(bpePath)
            specialTokens = ['|'] + CLASSLIST
            bpeWrapper.add_special_tokens({"additional_special_tokens": specialTokens})
            self.tokenizeModel = bpeWrapper
            self.vocabSize = len(self.tokenizeModel.vocab)
            self.vocab = self.tokenizeModel.vocab
            self.stride = 5

        elif self.tokenization == "base":
            self.vocabSize = len(BASEDICT)
            self.vocab = BASEDICT

        self.minSeqLen = self.minSeqLen * max(self.stride, 1)
        self.maxSeqLen = self.maxSeqLen * self.stride if self.stride > 1 else self.maxSeqLen
        self.extraTokenId = set([self.vocab[_] for _ in EXTRASET])

        self.data = None
        self.dataIdx = None
        self.classNum = 0
    
    def load(self):
        raise NotImplementedError()
    
    def process(self):
        raise NotImplementedError()


class PretrainDataset(BaseDataset):
    def __init__(self, dataPath: str, mode: str, args: argparse.Namespace):
        super().__init__(dataPath, mode, args)

        self.maxToken = args.maxToken
        self.maskRate = args.maskRate
        self.species = [] if args.species == "" else args.species.split(',')
        self.randomLenRate = args.randomLenRate
        self.speciesClassification = args.speciesClassification
        self.seqLenWarmup = args.seqLenWarmup

        self.speciesIdx = []
        self.classDict = {}

        self.load()
        self.getIndices()
    
    def load(self):
        self.data = []
        print(f"Generating pre-training data from {self.dataPath}...\n")
        if self.dataPath.endswith(".fa") or self.dataPath.endswith(".fasta") or self.dataPath.endswith(".fna"):
            rawData = pysam.FastaFile(self.dataPath)
            allSpecies = {}

            if self.speciesClassification:
                for ref, refsize in zip(rawData.references, rawData.lengths):
                    _ref = ref.split('|')
                    _ref = _ref[2] + '|' + _ref[3].split('_')[0] if self.trainMode == "pretrain" else _ref[2]
                    
                    if _ref not in allSpecies: allSpecies[_ref] = refsize
                    else: allSpecies[_ref] = allSpecies[_ref] + refsize
                
                allSpecies = [(k, v) for k, v in allSpecies.items()]
                allSpecies = sorted(allSpecies, key=lambda x: x[-1], reverse=True)
                allSpecies = {k[0]:i for (i, k) in enumerate(allSpecies)}
                print(f"Get {len(allSpecies)} species from pretraining data, the raw header num is {len(rawData.references)}\n")

            for i, ref in enumerate(rawData.references):
                if len(self.species) > 0 and ref not in self.species: continue
                chrom = rawData[ref].upper().replace("N", "")
                if self.trainMode == "pretrain" and len(chrom) < self.maxSeqLen: continue

                self.data.append(chrom)
                # if self.speciesClassification: self.speciesIdx.append(ref.split('|')[0])
                
                if self.speciesClassification: 
                    _ref = ref.split('|')
                    _ref = f"{_ref[2]}|{_ref[3].split('_')[0]}" if self.trainMode == "pretrain" else _ref[2]
                    self.speciesIdx.append(allSpecies[_ref])
            
            self.classDict = allSpecies
            rawData.close()
            self.classNum = len(set(self.speciesIdx))
            self.data = np.array(self.data, dtype=object)
        
        else:
            accumSeq = []
            accumSeqLen = 0
            with open(self.dataPath, "r") as f:
                for line_id, line in enumerate(f):
                    seq = line.strip().upper().replace("N", "")
                    accumSeq.append(seq)
                    accumSeqLen += len(seq)

                    if accumSeqLen >= self.loadChunk:
                        __seq = "".join(accumSeq)
                        self.data.append(__seq[:self.loadChunk])
                        accumSeq = [__seq[self.loadChunk:]]
                        accumSeqLen = len(__seq[self.loadChunk:])
                        print(f"\rFinish loading {len(self.data)} chunks...")

                if len(accumSeq) >= self.loadChunk or len(self.data) == 0:
                    self.data.append("".join(accumSeq))
                    accumSeq = []
                    accumSeqLen = 0
            
            self.data = np.array(self.data, dtype=object)
    
    def getIndices(self):
        assert len(self.data) > 0

        indices = []

        if self.trainMode == "pretrain":
            for i, seq in enumerate(self.data):
                startPos = random.randint(0, min(self.maxSeqLen-1, max(0, len(seq) - self.maxSeqLen)))
                for pos in range(startPos, len(seq)-self.maxSeqLen, self.maxSeqLen):
                    seqLen = self.maxSeqLen
                    if self.randomLenRate > 0 and self.mode == "train" and self._step >= self.seqLenWarmup:
                        if random.random() < self.randomLenRate: seqLen = random.randint(self.minSeqLen, self.maxSeqLen)

                    indices.append([i, pos, seqLen, self.speciesIdx[i] if len(self.speciesIdx) > 0 else -1])
        else:
            for i, seq in enumerate(self.data):
                indices.append([i, 0, len(seq), self.speciesIdx[i] if len(self.speciesIdx) > 0 else -1])

        self.dataIdx = np.array(indices, dtype=np.int64)
        print(f"Finish generating indices for {len(self.dataIdx)} samples...\n")
    
    def __processMLM(self, indices: list[int]) -> PackedData:
        accumLen = 0
        globalSeqArray = []
        globalPoses = []
        globalRaw = []
        speciesList = []

        # frag_id, start_pos, length, species_id
        for i, _idx in enumerate(indices):
            (fragID, startPos, fragLen, speciesID) = self.dataIdx[_idx].tolist()
            if self.seqLenWarmup > 0:
                fragLen = self.minSeqLen + int((self.maxSeqLen - self.minSeqLen) * min(1, self._step / self.seqLenWarmup))

            seq = self.data[fragID][startPos:startPos+fragLen]
            seq = reverse_complement(seq, apply=random.random() < 0.5 if self.augment and self.mode == "train" else False)
            seq = tokenizing(seq, self.tokenization, self.vocab, self.k, self.stride, ["<s>"], ["</s>"], self.tokenizeModel)
            if accumLen + len(seq) > self.maxToken: break

            if self.stride > 0 and self.stride < self.k:
                overlap_range = list(range(1+self.k, len(seq)-1-self.k, self.k * 2))
                poses = random.sample(overlap_range, int(self.maskRate*len(overlap_range)))
                poses = sorted(poses)
                tmp_pos = []
                for pos in poses:
                    tmp_pos.extend(list(range(pos - self.k, pos + self.k)))
                poses = tmp_pos
            else:
                poses = random.sample(range(1, len(seq)-1), int(self.maskRate*(len(seq)-2)))
                poses = sorted(poses)

            raw = []
            if self.speciesClassification and speciesID >= 0:
                raw.append(self.vocab[f"<class_{speciesID}>"])

            for pos in poses:
                raw.append(seq[pos])
                if seq[pos] == self.vocab['|']: continue

                mlmMethod = random.choices(range(3), k=1, weights=[0.8, 0.1, 0.1])[0]
                if mlmMethod == 0: seq[pos] = self.vocab["<mask>"]
                elif mlmMethod == 1:
                    tmpIdx = seq[pos]
                    while tmpIdx == seq[pos] or tmpIdx in self.extraTokenId:
                        tmpIdx = random.randint(0, len(self.vocab) - 1)
            
            raw = np.array(raw, dtype=np.int64)
            if self.speciesClassification and speciesID >= 0: poses = [0] + poses
            poses = np.array(poses, dtype=np.int64)
            accumLen += len(seq)

            # concat to global
            globalSeqArray.append(deepcopy(seq))
            globalPoses.append(poses)
            globalRaw.append(raw)
            if self.speciesClassification and speciesID >= 0: speciesList.append(self.vocab[f"<class_{speciesID}>"])
        
        globalRaw = np.concatenate(globalRaw, axis=0)
        speciesList = np.array(speciesList, dtype=np.int64)
        globalSeqArray = padding(globalSeqArray, self.vocab)
        globalSeqArray = np.stack(globalSeqArray, axis=0)

        for i, offset in enumerate(range(0, len(globalSeqArray) * globalSeqArray.shape[-1], globalSeqArray.shape[-1])):
            globalPoses[i] += offset

        globalPoses = np.concatenate(globalPoses, axis=0)

        self._step += self.numWorker
        return PackedData(globalSeqArray, globalRaw, globalPoses, None if np.any(speciesList == -1) or self.classNum < 2 else speciesList, augment=self.augment)
    
    def __processNTP(self, indices: list[int]) -> PackedData:
        accumLen = 0
        globalSeqArray = []
        speciesList = []

        # frag_id, start_pos, length, species_id
        for i, _idx in enumerate(indices):
            (fragID, startPos, fragLen, speciesID) = self.dataIdx[_idx].tolist()
            if self.seqLenWarmup > 0:
                fragLen = self.minSeqLen + int((self.maxSeqLen - self.minSeqLen) * min(1, self._step / self.seqLenWarmup))

            fragLen -= 2
            if self.speciesClassification: fragLen -= 1
            seq = self.data[fragID][startPos:startPos+fragLen]
            seq = reverse_complement(seq, apply=random.random() < 0.5 if self.augment and self.mode == "train" else False)

            _eos = ["</s>"]
            if self.speciesClassification and speciesID >= 0: _eos.append(f"<class_{speciesID}>")
            seq = tokenizing(seq, self.tokenization, self.vocab, self.k, self.stride, ["<s>"], _eos, self.tokenizeModel)
            if accumLen + len(seq) > self.maxToken: break

            accumLen += len(seq)

            # concat to global
            globalSeqArray.append(deepcopy(seq))
            if self.speciesClassification and speciesID >= 0: speciesList.append(self.vocab[f"<class_{speciesID}>"])
        
        globalSeqArray = padding(globalSeqArray, self.vocab)
        globalSeqArray: np.ndarray = np.stack(globalSeqArray, axis=0)
        globalRaw = globalSeqArray[:, 1:].reshape(-1)
        speciesList = np.array(speciesList, dtype=np.int64)
        
        self._step += self.numWorker
        return PackedData(globalSeqArray, globalRaw, None, None if np.any(speciesList == -1) or self.classNum < 2 else speciesList, augment=self.augment)
    
    def __processEMB(self, indices: list[int]) -> PackedData:
        globalSeqArray = []
        speciesList = []

        # frag_id, start_pos, length, species_id
        for i, _idx in enumerate(indices):
            (fragID, startPos, fragLen, speciesID) = self.dataIdx[_idx].tolist()

            seq = self.data[fragID][startPos:startPos+fragLen]
            seq = tokenizing(seq, self.tokenization, self.vocab, self.k, self.stride, ["<s>"], ["</s>"], self.tokenizeModel)

            # concat to global
            globalSeqArray.append(deepcopy(seq))
            if self.speciesClassification and speciesID >= 0: speciesList.append(speciesID)
        
        globalSeqArray = padding(globalSeqArray, self.vocab)
        globalSeqArray: np.ndarray = np.stack(globalSeqArray, axis=0)
        speciesList = np.array(speciesList, dtype=np.int64)
        
        return PackedData(globalSeqArray, speciesList)
    
    def process(self, indices: list[int]) -> PackedData:
        if self.trainMode == "pretrain":
            if self.pretrainTask == "MLM": return self.__processMLM(indices)
            elif self.pretrainTask == "NTP": return self.__processNTP(indices)
        elif self.finetuneTask == "embedding": return self.__processEMB(indices)
        else: raise ValueError("Invalid pretrainTask")
    
    def __len__(self) -> int:
        return len(self.dataIdx)
    
    def __getitem__(self, index: int):
        return index

class PretrainDataModule(LightningDataModule):
    def __init__(self, dataPath: str, args: argparse.Namespace):
        super(PretrainDataModule, self).__init__()

        self.dataPath = dataPath

        if args.trainMode == "pretrain":
            self.data = {
                "train": PretrainDataset(self.dataPath, "train", args),
                "eval": PretrainDataset(self.dataPath + ".eval", "eval", args)
            }
            self.vocab = self.data["train"].vocab
        else:
            self.data = {
                "predict": PretrainDataset(self.dataPath, "eval", args),
            }
            self.vocab = self.data["predict"].vocab
        
        self.batchSize = args.batchSize
        self.args = args
        self._step = 0

    def setup(self, stage):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.data["train"], shuffle=True,
            batch_size=self.args.batchSize if self.args.seqLenWarmup >= self._step else self.args.maxToken // self.args.maxSeqLen,
            collate_fn=self.data["train"].process,
            num_workers=self.args.numWorker,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.data["eval"],
            batch_size=self.args.batchSize,
            collate_fn=self.data["eval"].process,
            num_workers=self.args.numWorker
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.data["eval"],
            batch_size=self.args.batchSize,
            collate_fn=self.data["eval"].process,
            num_workers=self.args.numWorker
        )
    
    def predict_dataloader(self):
        return DataLoader(
            self.data["predict"], shuffle=False,
            batch_size=self.args.batchSize,
            collate_fn=self.data["predict"].process,
            num_workers=self.args.numWorker,
            pin_memory=True
        )


class FinetuneDataset(BaseDataset):
    def __init__(self, dataPath: str, mode: str, args: argparse.Namespace):
        super().__init__(dataPath, mode, args)

        self.load()
    
    def load(self):
        data = []
        mode = self.mode
        if self.mode == "eval": mode = "dev"

        rawData = None
        _pathList = glob.glob(self.dataPath + f"/{mode}.*")
        for _path in _pathList:
            if ".csv" in _path:
                rawData = pd.read_csv(_path)
                break
            elif ".parquet" in _path:
                rawData = pd.read_parquet(_path)
                break
        
        slice_idx = 1
        if "EPI" in self.dataPath: slice_idx = -1

        __label = {}
        for row in rawData.itertuples(index=False):
            seqs = []
            for _ in row[:slice_idx]:
                seq = re.sub(r"[^ATCG]", "", _.strip().upper())
                seqs.append(seq)
            data.append(tuple([*seqs, row[-1]]))
            if row[-1] not in __label: __label[row[-1]] = 0
            else: __label[row[-1]] += 1
        
        self.classNum = len(__label)
        self.data = deepcopy(data)
        del data
    
    def __processClassification(self, indices: list[int]) -> PackedData:
        globalSeqArray = []
        globalLabel = []

        for rev in range(2):
            if rev > 0:
                if not (self.augment and self.mode != "train"): break
            for idx in indices:
                data: tuple = self.data[idx]
                seq = None
                for col in range(len(data)-1):
                    __seq = data[col]
                    if self.augment: __seq = reverse_complement(__seq, apply=random.random() < 0.5 if self.mode == "train" else rev > 0)
                    __seq = tokenizing(__seq, self.tokenization, self.vocab, self.k, self.stride, ["<s>"] if col == 0 else ["|"], ["|"] if col < len(data)-2 else ["</s>"], self.tokenizeModel)
                    seq = np.concatenate([seq, __seq], axis=0) if seq is not None else __seq

                globalSeqArray.append(seq)
                if rev == 0: globalLabel.append(self.vocab[f"<class_{data[-1]}>"])
        
        globalSeqArray = padding(globalSeqArray, self.vocab)
        globalSeqArray: np.ndarray = np.stack(globalSeqArray, axis=0)

        return PackedData(globalSeqArray, globalLabel, task="classification", promptIdx=np.full((globalSeqArray.shape[0],), 0, dtype=np.int64), augment=self.augment)
    
    def __processEPIGeneration(self, indices: list[int]) -> PackedData:
        E2P = True if self.args.epiTask == "E2P" else False

        globalSrc, globalTgt, globalLabel = [], [], []
        for idx in indices:
            data: tuple = self.data[idx]
            if len(data) != 3: continue

            if E2P: src, tgt, typeid = data
            else: tgt, src, typeid = data

            src = tokenizing(src, self.tokenization, self.vocab, self.k, self.stride, ["<s>"], ["</s>"], self.tokenizeModel)
            tgt = tokenizing(tgt, self.tokenization, self.vocab, self.k, self.stride, ["<s>"], ["</s>"], self.tokenizeModel)

            globalSrc.append(src)
            globalTgt.append(tgt)
            globalLabel.append(typeid)
        
        globalSrc = np.stack(padding(globalSrc, self.vocab), axis=0)
        globalTgt = np.stack(padding(globalTgt, self.vocab), axis=0)

        return PackedData(globalSrc, globalTgt, promptIdx=globalLabel, task="generation")

    def process(self, indices: list[int]) -> PackedData:
        if self.finetuneTask == "classification": return self.__processClassification(indices)
        elif self.finetuneTask == "generation":
            if "EPI_generation" in self.dataPath: return self.__processEPIGeneration(indices)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index):
        return index

class FinetuneDataModule(LightningDataModule):
    def __init__(self, dataPath: str, args: argparse.Namespace):
        super(FinetuneDataModule, self).__init__()

        self.dataPath = dataPath
        self.data = {
            "train": FinetuneDataset(self.dataPath, "train", args),
            "eval": FinetuneDataset(self.dataPath, "eval", args),
            "test": FinetuneDataset(self.dataPath, "test", args)
        }
        self.vocab = self.data["train"].vocab
        
        self.batchSize = args.batchSize
        self.args = args

    def setup(self, stage):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.data["train"], shuffle=True,
            batch_size=self.args.batchSize,
            collate_fn=self.data["train"].process,
            num_workers=self.args.numWorker,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.data["eval"],
            batch_size=self.args.generateBatchSize,
            collate_fn=self.data["eval"].process,
            num_workers=self.args.numWorker
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.data["test"],
            batch_size=self.args.generateBatchSize,
            collate_fn=self.data["test"].process,
            num_workers=self.args.numWorker
        )
        
