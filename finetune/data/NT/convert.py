import os
import glob
import pysam
import pandas as pd

def parser(dataDir: str) -> list[tuple]:
    data = []
    with open(dataDir, 'r') as f:
        cache = []
        head = []
        for line in f:
            if line.startswith('>'):
                line = line[1:]
                _chr = line.strip().split(':')[0]
                _label = int(line.strip().split('|')[1])
                if len(cache) == 0:
                    head.extend([_chr, _label])
                else:
                    data.append((*head, ''.join(cache)))
                    cache.clear()
                    head.clear()
                    head.extend([_chr, _label])
            else:
                cache.append(line.strip().upper())
        
        if len(cache) > 0 and len(head) > 0:
            data.append((*head, ''.join(cache)))
    
    return data

def list2pd(data: list[tuple]) -> dict:
    seq, label = [], []
    for _chr, _label, _seq in data:
        seq.append(_seq)
        label.append(_label)
    
    return {"sequence": seq, "label": label}

def convert(baseDir):
    trainDir = os.path.join(baseDir, "train.fna")
    testDir = os.path.join(baseDir, "test.fna")

    trainval, test = parser(trainDir), parser(testDir)
    
    train, val = [], []
    for data in trainval:
        if data[0] == "chr10": val.append(data)
        else: train.append(data)
    
    for data, split in zip([train, val, test], ["train", "dev", "test"]):
        df = pd.DataFrame(list2pd(data))
        df.to_csv(os.path.join(baseDir, f"{split}.csv"), sep=',', index=False)


if __name__ == "__main__":
    curDir = os.path.dirname(os.path.abspath(__file__))
    
    default_val = 'chr10'

    _tgt_dir = curDir + "/*"
    _list = glob.glob(_tgt_dir, recursive=True)
    _list = [_ for _ in _list if os.path.isdir(_)]

    for _file in _list:
        dataName = os.path.basename(_file)
        trainDir, testDir = os.path.join(_file, "train.fna"), os.path.join(_file, "test.fna")

        trainval, test = parser(trainDir), parser(testDir)
        train, val = [], []
        for data in trainval:
            if data[0] == "chr10": val.append(data)
            else: train.append(data)
        
        testChr = ','.join(set([_[0] for _ in test]))
        print(f"{dataName} test: {testChr}|{len(test)}, val: chr10|{len(val)}, train: {len(train)}")
        
        for data, split in zip([train, val, test], ["train", "dev", "test"]):
            df = pd.DataFrame(list2pd(data))
            df.to_csv(os.path.join(_file, f"{split}.csv"), sep=',', index=False)