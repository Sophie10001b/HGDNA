import os
import pandas as pd
import random
import glob
import re
import pysam
import subprocess

LABELDICT = {"human": 0, "lemur": 1, "mouse": 2, "pig": 3, "hippo": 4}
LABELHEAD = ",".join([f"{k}:{v}" for k, v in LABELDICT.items()])

def main(train: list[str], val: list[str], test: list[str], trainSize: int, valSize: int, testSize: int, seqLen: int, needFasta: bool):
    curDir = os.path.dirname(os.path.abspath(__file__))
    chrList = glob.glob(curDir + "/*/*.fna.gz", recursive=True)
    resDir = os.path.join(curDir, f"species_{seqLen}_{trainSize}_{valSize}_{testSize}")

    if not os.path.exists(resDir): os.mkdir(resDir)
    
    trainCache, valCache, testCache = {}, {}, {}
    trainData, valData, testData = {}, {}, {}

    for chrPath in chrList:
        _chr = os.path.basename(chrPath).split(".")[0]
        if _chr not in train + val + test: continue

        if chrPath.endswith(".gz"):
            _cmd = f"gzip -d -k {chrPath}"
            subprocess.run(_cmd, stdout=subprocess.PIPE, shell=True)
            chrPath = chrPath.replace(".fna.gz", ".fna")

        data = pysam.FastaFile(chrPath)
        species = os.path.basename(os.path.dirname(chrPath))

        chorm = data[data.references[0]].strip().upper()
        chorm = re.sub(r'[^ACGT]', '', chorm)
        if _chr in train: trainCache[f"0|0|{species}|{_chr}"] = chorm
        elif _chr in val: valCache[f"0|0|{species}|{_chr}"] = chorm
        elif _chr in test: testCache[f"0|0|{species}|{_chr}"] = chorm
    
    for (cache, finalData, datasize) in zip([trainCache, valCache, testCache], [trainData, valData, testData], [trainSize, valSize, testSize]):
        _sum = sum([len(_) for _ in cache.values()])
        for _chr in cache.keys():
            _num_seq = int((len(cache[_chr]) / _sum) * datasize)
            
            startIdx = random.sample(range(len(cache[_chr]) - seqLen), k=_num_seq)

            for i, _idx in enumerate(startIdx): finalData[f"{_chr}|{i}"] = cache[_chr][_idx: _idx + seqLen]
    
    # write .fa and parquet
    for data, split in zip([trainData, valData, testData], ['train', 'dev', 'test']):
        if needFasta:
            with open(os.path.join(resDir, f"{split}.fa"), 'w') as f:
                for k, v in data.items(): f.write(f">{k}\n{v}\n")
        
        # write parquet
        df = [(v, LABELDICT[k.split('|')[2]]) for k, v in data.items()]
        df = pd.DataFrame(df, columns=["sequence", f"label, {LABELHEAD}"])
        df.to_parquet(os.path.join(resDir, f"{split}.parquet"))
        
        if os.path.exists(os.path.join(curDir, f"{split}.fa.fai")): os.remove(os.path.join(curDir, f"{split}.fa.fai"))
    
    _cmd = f"rm -rf ./**/*.fna ./**/*.fna.fai"
    subprocess.run(_cmd, stdout=subprocess.PIPE, shell=True)
        

if __name__ == "__main__":
    random.seed(17)

    train = [f'chr{i}' for i in range(1, 10)]
    val = ['chr10']
    test = ['chr11']
    trainSize, valSize, testSize = 10000, 1000, 1000

    for seqLen in [1024, 16384, 32768]: main(train, val, test, trainSize, valSize, testSize, seqLen, False)