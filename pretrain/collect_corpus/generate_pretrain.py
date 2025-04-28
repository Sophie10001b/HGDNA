import os
import re
import glob
import numpy as np
import pysam
import pandas as pd
import subprocess
import multiprocessing

from tqdm.std import tqdm

FILTER = 1024
CHUNK = 16384
SPECIES = ["fungi", "protozoa", "vertebrate_mammalian", "invertebrate", "vertebrate_other", "bacteria", "plant"]
COMPDICT = {'A':'T', 'T':'A', 'C':'G', 'G':'C', 'N': 'N'}

def seqClean(seq: str) -> str:
    seq = seq.strip().upper()
    seq = re.sub(r"[^ATCG]", 'N', seq)
    return seq

def reverse_complete(seq: str) -> str:
    reverse_seq = ''.join(COMPDICT[_] for _ in reversed(seq))
    return reverse_seq

def dataProcess(info: tuple) -> list:
    (subSpecies_id, subSpeciesDir, species_id, species, window, onlyAnnotation, targetRegin) = info

    _genomeCompressDir = glob.glob(os.path.join(subSpeciesDir, '*.fna.gz'), recursive=True)
    _annotationCompressDir = glob.glob(os.path.join(subSpeciesDir, '*.gff.gz'), recursive=True)

    for _path in _genomeCompressDir:
        _cmd = f"gzip -d {_path}"
        subprocess.run(_cmd, stdout=subprocess.PIPE, shell=True)
    for _path in _annotationCompressDir:
        _cmd = f"gzip -d {_path}"
        subprocess.run(_cmd, stdout=subprocess.PIPE, shell=True)

    _subSpecies_name = os.path.basename(subSpeciesDir)
    _genomeDir = glob.glob(os.path.join(subSpeciesDir, '*.fna'), recursive=True)
    _annotataionDir = glob.glob(os.path.join(subSpeciesDir, '*.gff'), recursive=True)

    if len(_genomeDir) == 0 or len(_annotataionDir) == 0:
        print(f"{subSpeciesDir} is lost\n")
        return ['', '', 0, 0, species, _subSpecies_name]

    _genomeDir = _genomeDir[0]
    _annotataionDir = _annotataionDir[0]
    _gemone_id = '_'.join(os.path.basename(_genomeDir).split('_')[:-1])

    prefix = f">{species_id}|{subSpecies_id}|{species.replace('|', '_').replace(' ', '_')}|{_subSpecies_name.replace('|', '_').replace(' ', '_')}|{_gemone_id.replace('|', '_').replace(' ', '_')}"

    seqFile = pysam.FastaFile(_genomeDir)
    all_count = 0
    annotation_count = 0
    seqs = []

    if onlyAnnotation:
        annotation = pd.read_csv(_annotataionDir, sep="\t", comment='#', header=None)
        annotation.columns = ["seqid", "source", "type", "start", "end", "score", "strand", "phase", "attributes"]

        annotationData = {}
        for i, row in annotation.iterrows():
            if row["type"] == "region": continue
            if targetRegin is not None and row["type"] not in targetRegin: continue
            _seqid, _start, _end = row["seqid"], int(row["start"]), int(row["end"])
            if _seqid not in annotationData: annotationData[_seqid] = []

            annotationData[_seqid].append((_start, _end))
        
        for key in annotationData.keys():
            annotationData[key] = list(set(annotationData[key]))

        for _i, _key in enumerate(seqFile.references):
            if seqFile.lengths[_i] < FILTER or _key not in annotationData: continue

            # seq = list(seqClean(seqFile[_key]))
            seq = np.array(list(seqClean(seqFile[_key])))
            all_count += len(seq)
            mask = np.zeros((len(seq,)), dtype=bool)

            tmp_seq = []

            for (_start, _end) in annotationData[_key]:
                # mask the annotated region
                _regin_len = _end - _start + 1
                if window < 1 and window > 0: window = (window * _regin_len) // 2
                _start_window = max(0, _start-1-window)
                _end_window = min(len(seq), _end+window)

                annotation_count += _regin_len
                tmp_seq.append(''.join(seq[_start_window:_end_window].tolist()))

                # mask[_start_window:_end_window] = True
            
            # seqs.append(''.join(seq[mask].tolist()))
            seqs.append('|'.join(tmp_seq))
        
        seqs = '|'.join(seqs)
        # annotation_count += len(seqs)
    
    else:
        for _i, _key in enumerate(seqFile.references):
            if seqFile.lengths[_i] < FILTER: continue

            seq = seqClean(seqFile[_key])
            all_count += len(seq)
            seqs.append(seq)
        
        seqs = '|'.join(seqs)
    
    seqFile.close()
    return [seqs, prefix, annotation_count, all_count, species, _subSpecies_name]

def getPretrain(logName: str, finalName: str, onlyAnnotation: bool = False, window: float=0, thread: int = 8, targetRegin: list=[]):
    curDir = os.path.dirname(os.path.abspath(__file__))
    species_count = 0
    all_length_all = 0
    all_length_annotation = 0

    if window >= 1: window = int(window)
    elif window > 0: window = float(window)
    else: window = 0

    logName = os.path.join(curDir, logName)
    finalName = os.path.join(curDir, finalName)
    # if not os.path.exists(finalName):
    with open(finalName, "w") as f:
        pass
    with open(logName, "w") as f:
        f.write(f"use annotation: {onlyAnnotation}\twindow: {window}\ttarget region: {','.join(targetRegin)}\n")

    targetRegin = set(targetRegin)

    for species_id, species in enumerate(SPECIES):
        speciesDir = os.path.join(curDir, species)
        if not os.path.exists(speciesDir): continue

        subSpeciesList = glob.glob(os.path.join(speciesDir, "*"))
        subSpeciesList = [_ for _ in subSpeciesList if os.path.isdir(_)]

        species_length_all = 0
        species_length_annotation = 0
        subSpeciesList = [(i, _, species_id, species, window, onlyAnnotation, targetRegin if len(targetRegin) > 0 else None) for i, _ in enumerate(subSpeciesList)]

        results = []
        # for _ in subSpeciesList: results.append(dataProcess(_))
        with multiprocessing.Pool(processes=thread) as pool:
            with tqdm(total=len(subSpeciesList), desc=f"Processing {species}...") as pbar:
                for _ in pool.imap_unordered(dataProcess, subSpeciesList):
                    results.append(_)
                    pbar.update()
            # results = pool.map(dataProcess, subSpeciesList)
        
        with open(finalName, 'a+') as _res_file:
            for (seq, prefix, annotation_count, all_count, species_name, subSpecies_name) in results:
                if seq == '':
                    print (f"{species_name}|{subSpecies_name} returns none...\n")
                    continue
                _res_file.write(prefix+'\n')
                seq_chunk = [seq[k:k+CHUNK] for k in range(0, len(seq), CHUNK)]
                _res_file.write('\n'.join(seq_chunk)+'\n')

                species_length_annotation += annotation_count
                species_length_all += all_count
        
                with open(logName, "a+") as _log_file:
                    _log_file.write(f"{species_name}|{subSpecies_name} = {all_count} (annotation: {annotation_count})\n")
        species_count += 1
        all_length_all += species_length_all
        all_length_annotation += species_length_annotation
    
        with open(logName, "a+") as _log_file:
            _log_file.write(f"========== {species} = {species_length_all} (annotation: {species_length_annotation}, rate: {species_length_annotation / species_length_all:.2f})\n")
    
    with open(logName, "a+") as _log_file:
        _log_file.write(f"========== ALL = {all_length_all} (annotation: {all_length_annotation}, rate: {all_length_annotation / all_length_all:.2f})\n")

if __name__ == "__main__":
    getPretrain("final_pretrain.log", "final_pretrain.fa", True, window=0, thread=16, targetRegin=['gene'])

    # curDir = os.path.dirname(os.path.realpath(__file__))
    # with open(os.path.join(curDir, "final_pretrain.fa"), 'r') as f:
    #     head = []
    #     length = []
    #     l = 0
    #     for line in f:
    #         if line[0] == '>':
    #             head.append(line.strip())
    #             if l > 0: length.append(l)
    #         else: l += len(line.strip())
        
    #     if l > 0: length.append(l)
    #     pass
    # a = pysam.FastaFile(os.path.join(curDir, "final_pretrain.fa"))
    # pass