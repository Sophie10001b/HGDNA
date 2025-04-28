import os
import math
import glob
import time
import ftputil
import pandas as pd
import random
import subprocess
import multiprocessing

from tqdm.std import tqdm
from ftplib import FTP
from ftputil.error import FTPIOError

# MB
MAXSIZE = {'fungi': 3000, 'protozoa': 3000, 'vertebrate_mammalian': 6000, 'invertebrate': 3000, 'vertebrate_other': 3000, 'bacteria': 3000, 'plant': 3000}

def subSpeciesMathch(raw: str, tgt: str, exact_match: bool=False) -> bool:
    tmpRaw = raw.lower().replace('_', ' ')
    tmpTgt = tgt.lower()

    if not exact_match:
        if tmpTgt in tmpRaw:
            return True
        elif tmpTgt.split(' ')[0] in tmpRaw:
            return True
        elif tmpTgt[:4] == tmpRaw[:4]:
            return True
        else: return False
    else:
        if tmpTgt == tmpRaw:
            return True
        else: return False

def _update_progress(data, bar: tqdm):
    bar.update(len(data))

def getDownloadUrl(savePath: str, clean: bool=False):
    curDir = os.path.dirname(os.path.abspath(__file__))
    targetCSV = pd.read_csv(os.path.join(curDir, "download_target.csv"), sep='\t')
    target = {}
    _finished = {}

    if not os.path.exists(savePath) or clean:
        with open(savePath, "w") as f:
            pass
    else:
        with open(savePath, "r") as f:
            for line in f:
                if line.startswith(">"):
                    finished_species = line[1:].strip().split('|')[0]
                    _finished[finished_species] = 1


    for key in targetCSV.columns:
        if key in _finished: continue
        _value = targetCSV[key].tolist()
        _value = [_.strip() for _ in _value if isinstance(_, str)]
        target[key] = _value

    ftpDir = "ftp.ncbi.nlm.nih.gov"
    baseDir = "/genomes/refseq"
    
    with ftputil.FTPHost(ftpDir, user="anonymous", passwd="@anonymous") as ftp_host:
        ftp_host.chdir(baseDir)
        ftp_host.keep_alive()
        
        for species in target.keys():
            final_target = []
            ftp_host.chdir(os.path.join(baseDir, species))

            # get assembly_summary.txt
            _summary = os.path.join(curDir, f'{species}_assembly_summary.txt')
            if not os.path.exists(_summary):
                _download_command = [
                    "wget",
                    f"ftp://ftp.ncbi.nlm.nih.gov/genomes/refseq/{species}/assembly_summary.txt",
                    "--show-progress",
                    "-O",
                    f"{species}_assembly_summary.txt"
                ]
                subprocess.run(_download_command, stdout=subprocess.PIPE)
                # ftp_host.download(os.path.join(ftp_host.getcwd(), "assembly_summary.txt"), _summary)

            cache_list = []

            _summary = pd.read_csv(_summary, sep='\t', skiprows=1)
            for _line in _summary.iterrows():
                if len(_line) > 1: _line = _line[-1]
                _ftp_path = _line['ftp_path']
                _ftp_path = _ftp_path[_ftp_path.find('gov/')+3:]
                _name = _line['organism_name'].replace(' ', '_')
                cache_list.append([_line['#assembly_accession'], _name, _ftp_path, _line['genome_size']])

            species_size = 0

            _cache = {}
            subSpecies_list = []
            if len(target[species]) > 0:
                _exact_match = True
                for tgtSubSpecies in target[species]:
                    for rawData in cache_list:
                        rawSubSpecies = rawData[1]
                        if rawSubSpecies not in _cache and subSpeciesMathch(rawSubSpecies, tgtSubSpecies, _exact_match):
                            subSpecies_list.append(rawData)
                            _cache[rawSubSpecies] = 1
            else:
                random.shuffle(cache_list)
                for rawData in cache_list:
                    rawSubSpecies = rawData[1]
                    subSpecies_list.append(rawData)
                    _cache[rawSubSpecies] = 1

            print(f">>>{species} matching {len(_cache)} results\n")

            for subSpeciesData in subSpecies_list:
                data_id, subSpecies, ftp_path, genome_size = subSpeciesData
                print(f"Start checking {species}|{subSpecies}|{data_id}, total size {species_size:.4f} Mb...\n")
                ftp_host.chdir(ftp_path)

                assembly_id = os.path.basename(ftp_path)
                genome = assembly_id + "_genomic.fna.gz"
                annotation = assembly_id + "_genomic.gff.gz"

                try:
                    if ftp_host.path.exists(genome) and ftp_host.path.exists(annotation):
                        _genome_size = float(genome_size) / (1000 * 1000)
                        if species_size + _genome_size >= MAXSIZE[species] * 1.5: continue
                        final_target.append([subSpecies, data_id, os.path.join(ftp_path, genome), os.path.join(ftp_path, annotation)])
                        species_size += _genome_size
                except (EOFError, FTPIOError) as e:
                    print(f"{species}|{subSpecies}|{data_id} failed to check, retry...\n")
                    if ftp_host.path.exists(genome) and ftp_host.path.exists(annotation):
                        _genome_size = float(genome_size) / (1000 * 1000)
                        if species_size + _genome_size >= MAXSIZE[species] * 1.5: continue
                        final_target.append([subSpecies, data_id, os.path.join(ftp_path, genome), os.path.join(ftp_path, annotation)])
                        species_size += _genome_size
                
                if species_size >= MAXSIZE[species]:
                    print(f"{species} exceeds maximum size {MAXSIZE[species]} Mb, skip...\n")
                    break
            
            with open(savePath, 'a+') as f:
                f.write(f">{species}|{species_size:.4f} Mb\n")
                for subSpeciesData in final_target:
                    if len(subSpeciesData) != 4: continue
                    (subSpecies, data_id, genome_url, annotation_url) = subSpeciesData
                    f.write(f"{subSpecies}\t{data_id}\t{genome_url}\t{annotation_url}\n")


def _download_subspecies(info: tuple):
    (_id, species, subSpecies, total, genome_url, local_genome_path, annotation_url, local_annotation_path) = info
    _show_progress = multiprocessing.current_process().name[-1] == '1'
    
    if not _show_progress:
        _genome_cmd = f"wget ftp://ftp.ncbi.nlm.nih.gov{genome_url} -P {os.path.dirname(local_genome_path)} --quiet && gzip -d {local_genome_path}"
        _annotation_cmd = f"wget ftp://ftp.ncbi.nlm.nih.gov{annotation_url} -P {os.path.dirname(local_annotation_path)} --quiet && gzip -d {local_annotation_path}"
    else:
        _genome_cmd = f"wget ftp://ftp.ncbi.nlm.nih.gov{genome_url} -P {os.path.dirname(local_genome_path)} --show-progress && gzip -d {local_genome_path}"
        _annotation_cmd = f"wget ftp://ftp.ncbi.nlm.nih.gov{annotation_url} -P {os.path.dirname(local_annotation_path)} --show-progress && gzip -d {local_annotation_path}"

    genome_success = 0
    annotation_success = 0
    if not os.path.exists(local_genome_path.replace('.gz', '')):
        subprocess.run(_genome_cmd, stdout=subprocess.PIPE, shell=True)
        genome_success = 1
    else:
        # print(f"{species}|{subSpecies} genome is already downloaded, skip...\n")
        genome_success = 1

    if not os.path.exists(local_annotation_path.replace('.gz', '')):
        subprocess.run(_annotation_cmd, stdout=subprocess.PIPE, shell=True)
        annotation_success = 1
    else:
        # print(f"{species}|{subSpecies} annotation is already downloaded, skip...\n")
        annotation_success = 1
    
    print(f">>>Finish downloading {species}|{subSpecies}, genome download: {_id+1}/{total}, annotation download: {_id+1}/{total}\n")
    
    return [genome_success, annotation_success]

def download(urlFile: str, thread: int = 8):
    curDir = os.path.dirname(os.path.abspath(__file__))
    urlDict = {}
    with open(urlFile, "r") as f:
        curSpecies = None
        for line in f:
            if line.startswith('>'):
                curSpecies = line[1:].strip().split('|')[0]
                urlDict[curSpecies] = []
            else:
                # subSpecies, data_id, genome_url, annotation_url
                urlDict[curSpecies].append(line.strip().split('\t'))
    
    ftpDir = "ftp.ncbi.nlm.nih.gov"
    baseftpDir = "/genomes/refseq"
    bar_fmt = "{desc} [{n:.2f} MB]"

    for species in urlDict.keys():
        speciesDir = os.path.join(curDir, species)
        if not os.path.exists(speciesDir):
            os.makedirs(speciesDir)
        
        task_list = []
        for _id, (subSpecies, data_id, genome_url, annotation_url) in enumerate(urlDict[species]):
            total = len(urlDict[species])
            subSpecies = subSpecies.replace(" ", "_") + '_' + data_id.replace(" ", "_")
            subSpeciesDir = os.path.join(speciesDir, subSpecies)

            if not os.path.exists(subSpeciesDir):
                os.makedirs(subSpeciesDir)

            local_genome_path = os.path.join(subSpeciesDir, os.path.basename(genome_url))
            local_annotation_path = os.path.join(subSpeciesDir, os.path.basename(annotation_url))

            task_list.append((_id, species, subSpecies, total, genome_url, local_genome_path, annotation_url, local_annotation_path))

        with multiprocessing.Pool(processes=thread) as pool:
            with tqdm(total=len(task_list), desc=f"Downloading {species}...") as pbar:
                for _ in pool.imap_unordered(_download_subspecies, task_list):
                    pbar.update()


if __name__ == "__main__":
    random.seed(17)
    curDir = os.path.dirname(os.path.abspath(__file__))
    savePath = os.path.join(curDir, "url.txt")
    getDownloadUrl(savePath)
    download(savePath)