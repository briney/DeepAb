#!/usr/bin/env python
#
# (c) Copyright Rosetta Commons Member Institutions.
# (c) This file is part of the Rosetta software suite and is made available under license.
# (c) The Rosetta software is developed by the contributing members of the Rosetta Commons.
# (c) For more information, see http://www.rosettacommons.org. Questions about this can be
# (c) addressed to University of Washington CoMotion, email: license@uw.edu.

## @file   create_antibody_db.py
## @brief  Script to create database for RosettaAntibody
## @author Jeliazko Jeliazkov

## @details download non-redundant Chothia Abs from SAbDab
## Abs are downloaded by html query (is there a better practice)?
## Abs are Chothia-numbered, though we use Kabat to define CDRs.
## After download, trim Abs to Fv and extract FR and CDR sequences.
## For the purpose of trimming, truncate the heavy @112 and light @109
## Some directories (antibody_database, info, etc...) are hard coded.


import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import os
from pathlib import Path
import requests
import sys
import time
import traceback

from os.path import basename

import pandas as pd

from Bio import SeqIO
from Bio.PDB import PDBParser, PDBIO
from Bio.SeqUtils import seq1

from bs4 import BeautifulSoup

from tqdm import tqdm

from abutils.io import list_files
from abutils.utils.alignment import local_alignment

from deepab.util.pdb import get_pdb_chain_seq



#---------------------------
#         SAbDab
#---------------------------


def download_sabdab_summary_file(summary_file,
                                 seqid=99,
                                 paired=True,
                                 nr_complex='All',
                                 nr_rfactor='',
                                 nr_res=4):
    '''
    Downloads a summary data file from the Structural Antibody Database 
    (`SAbDab`_) [Schneider22]_.

    .. seealso::
       | C Schneider, MIJ Raybould, CM Deane 
       | SAbDab in the Age of Biotherapeutics: Updates including SAbDab-Nano, the Nanobody Structure Tracker  
       | *Nucleic Acids Research* 2022. https://doi.org/10.1093/nar/gkab1050  

    Parameters
    ----------
    summary_file : str
        Path to which the summary file data will be written, in tab-delimited 
        format. Directories will be created if they do not exist. Required.

    seqid : int or float, default=99
        Sequence identity at which to group antibodies from SAbDab. Only one 
        sequences from each group will be downloaded. 

    paired : bool, default=True
        If ``True``, only sequences with paired H/L chains will be downloaded. 

    nr_complex : str, default='All'
        Types of structural complexes to download. Options are ``'All'``, 
        ``'Bound'``, or ``'Unbound'``.

    nr_rfactor : str, default=''
        R-factor cutoff. If not provided, no cutoff is used.

    nr_res : int or float, default=4
        Minimum resolution (in angstroms) of downloaded structures.

    .. _SAbDab: http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/search/
    '''
    base_url = 'http://opig.stats.ox.ac.uk'
    search_url = base_url + '/webapps/newsabdab/sabdab/search/'
    params = dict(seqid=seqid,
                  paired=paired,
                  nr_complex=nr_complex,
                  nr_rfactor=nr_rfactor,
                  nr_res=nr_res)
    query = requests.get(search_url, params=params)
    html = BeautifulSoup(query.content, 'html.parser')
    summary_url = base_url + html.find(id='downloads').find('a').get('href')
    download_file(summary_url, summary_file)
        
        
def parse_sabdab_summary(file_name):
    """
    Parses SAbDab summary file

    Parameters
    ----------
    file_name : str
        Path to a SAbDab summary file, in tab-delimited format.

    Returns
    -------
    sabdab_dict : dict
        A dictionary containing SAbDab summary data, with the format:
        ``sabdab_dict[pdb_id] = {column1: value1, column2: value2 ...}``
    """
    # dict[pdb] = {col1 : value, col2: value, ...}
    sabdab_dict = {}

    with open(file_name, "r") as f:
        # first line is the header, or all the keys in our sub-dict
        header = f.readline().strip().split("\t")
        # next lines are data
        for line in f.readlines():
            split_line = line.strip().split("\t")
            td = {}  # temporary dict of key value pairs for one pdb
            for k, v in zip(header[1:], split_line[1:]):
                # pdb id is first, so we skip that for now
                td[k] = v
            # add temporary dict to sabdab dict at the pdb id
            sabdab_dict[split_line[0]] = td
    return sabdab_dict



#---------------------------
#          PDBs
#---------------------------


def download_pdb_files(pdb_ids,
                       download_path,
                       max_workers=16,
                       logfile=None):
    """
    :param pdb_ids: A set of PDB IDs to download
    :type pdb_ids: set(str)
    :param antibody_database_path: Path to the directory to save the PDB files to.
    :type antibody_database_path: str
    :param max_workers: Max number of workers in the thread pool while downloading.
    :type max_workers: int
    """
    print('\ndownloading PDB files:')
    sys.stdout.flush()
    pdb_files = [os.path.join(download_path, pdb + '.pdb') for pdb in pdb_ids]
    download_url = 'http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/pdb/{}/?scheme=chothia'
    urls = [download_url.format(pdb) for pdb in pdb_ids]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = [
            executor.submit(lambda a: download_file(*a), args)
            for args in zip(urls, pdb_files)
        ]
        for _ in tqdm(as_completed(results), total=len(urls)):
            pass
    downloaded_files = list_files(download_path)
    
    # logging
    log_info = []
    n_downloads = len(downloaded_files)
    n_pdbs = len(pdb_ids)
    i = f"{n_downloads} of {n_pdbs} PDB IDs were successfully downloaded"
    print(i)
    sys.stdout.flush()
    log_info.append(i)
    if n_pdbs > n_downloads:
        failed = []
        for pdb_id in pdb_ids:
            if not any([pdb_id in df for df in downloaded_files]):
                failed.append(pdb_id)
        log_info.append('The following PDB IDs were not downloaded:')
        log_info.extend(failed)
    if logfile is not None:
        write_log(logfile, '\n'.join(log_info))
        
        
def process_pdbs(raw_dir,
                 processed_dir,
                 sabdab_summary_file,
                 ignore_single_chains=True,
                 logfile=None):
    """

    """
    print('\nprocessing pdb files:')
    sys.stdout.flush()
    pdb_warnings = []
    same_chains = []
    sabdab_missing = []
    
    # Parse the SAbDab summary file
    sabdab_dict = parse_sabdab_summary(sabdab_summary_file)

    pdb_files = [f for f in os.listdir(raw_dir) if f.endswith(".pdb")]
    unique_pdb_ids = set([p[:4] for p in pdb_files])

    for pdb_id in tqdm(unique_pdb_ids):
        if pdb_id not in sabdab_dict:
            pdb_file = os.path.join(raw_dir, f'{pdb_id}.pdb')
            chain_info = parse_pdb_chain_names(pdb_file)
            if chain_info is None:
                sabdab_missing.append(pdb_id)
                continue
            else:
                sabdab_dict[pdb_id] = chain_info
        warn, same_chain = process_pdb(
                pdb_id,
                raw_dir,
                processed_dir,
                sabdab_dict,
                ignore_single_chains=ignore_single_chains
        )
        if warn is not None:
            pdb_warnings.append(warn)
        if same_chain is not None:
            same_chains.append(same_chain)
    
    # logging
    log_info = []
    if same_chains:
        i = f"Removed a total of {len(same_chains)} PDBs with same chain VH/VLs"
        print(i)
        sys.stdout.flush()
        log_info.append(i + ':')
        log_info.append(f"\t{', '.join(same_chains)}\n")
    if sabdab_missing:
        i = f'{len(sabdab_missing)} PDB IDs were not found in the SAbDab summary file'
        print(i)
        sys.stdout.flush()
        log_info.append(i)
    else:
        i = 'All PDB IDs were found in the SAbDab summary file.'
        log_info.append(i)
    if pdb_warnings:
        i = f"{len(pdb_warnings)} PDB files had errors during processing"
        print(i)
        sys.stdout.flush()
        log_info.append(i + ':')
        warns = '\n'.join(pdb_warnings)
        log_info.append(f"{warns}\n")
    if logfile is not None:
        write_log(logfile, '\n'.join(log_info))


def process_pdb(pdb_id,
                raw_dir,
                processed_dir,
                sabdab_dict,
                ignore_single_chains):
    """
    :param pdb_id: The PDB ID of the protein
    :param antibody_database_path:
        The path where all the chothia numbered pdb files are stored as [pdb_id].pdb
    :param ignore_same_VL_VH_chains:
        Whether or not to ignore a PDB file when its VL chain and VH chains are
    :type ignore_single_chains: bool
    """
    # if an old processed PDB file already exists, delete it
    raw_pdb = os.path.join(raw_dir, pdb_id + '.pdb')
    processed_pdb = os.path.join(processed_dir, pdb_id + '.pdb')
    if os.path.isfile(processed_pdb):
        os.remove(processed_pdb)
    
    # read the raw PDB file
    if not os.path.isfile(raw_pdb):
        err = f'MISSING FILE: {pdb_id}'
        return (err, None)
    with open(raw_pdb, 'r') as f:
        pdb_text = f.read()
    if not pdb_text:
        err = f'EMPTY FILE: {pdb_id}'
        return (err, None)

    # get hchain and lchain data from the SAbDab summary file, if givenj
    hchain_text, lchain_text = '', ''
    hchain = sabdab_dict[pdb_id]["Hchain"]
    lchain = sabdab_dict[pdb_id]["Lchain"]

    # we do not currently have a good way of handling VH & VL on the same chain
    if ignore_single_chains and hchain == lchain:
        err = f'SINGLE CHAIN: {pdb_id}'
        return (None, err)
    
    # skip PDBs with missing HC or LC (we want paired only)
    if any([hchain == 'NA', lchain == 'NA']):
        na = []
        if hchain == 'NA':
            na.append('H')
        if lchain == 'NA':
            na.append('L')
        s = 'S' if len(na) > 1 else ''
        err = f"NA {'/'.join(na)} CHAIN{s}: {pdb_id}"
        return (err, None)

    # process the PDB data for H and L chains
    hchain_text = process_chain(pdb_text, hchain, 112, 'H')
    if not hchain_text:
        err = f'NO H CHAIN TEXT: {pdb_id}'
        return (err, None)
    lchain_text = process_chain(pdb_text, lchain, 109, 'L')
    if not lchain_text:
        err = f'NO L CHAIN TEXT: {pdb_id}'
        return (err, None)
    
    # write the processed PDB data
    with open(processed_pdb, 'w') as f:
        f.write(hchain_text + lchain_text)
    return (None, None)


def parse_pdb_chain_names(pdb_file):
    """Gets the heavy and light chain ID's from a chothia file from SAbDab"""
    # Get the line with the HCHAIN and LCHAIN
    hl_line = ''
    with open(pdb_file) as f:
        for line in f.readlines():
            if 'PAIRED_HL' in line:
                hl_line = line
                break
    if hl_line == '':
        return None

    words = hl_line.split(' ')
    h_chain = l_chain = None
    for word in words:
        if word.startswith('HCHAIN'):
            h_chain = word.split('=')[1]
        if word.startswith('LCHAIN'):
            l_chain = word.split('=')[1]
    return {'Hchain': h_chain, 'Lchain': l_chain}
    
    
def process_chain(pdb_text, chain, resnum, newchain):
    """
    Read PDB line by line and return all lines for a chain,
    with a resnum less than or equal to the input.
    This has to be permissive for insertion codes.
    This will return only a single truncated chain.
    This function can update chain to newchain.
    """
    trunc_text = ""
    for line in pdb_text.split("\n"):
        if (line.startswith("ATOM") 
            and line[21] == chain
            and int(line[22:26]) <= resnum
           ):
            trunc_text += line[:21] + newchain + line[22:]
            trunc_text += "\n"
    return trunc_text



#---------------------------
#         FASTAs
#---------------------------


def download_fasta_files(pdb_ids, download_dir, max_workers=16, sleep=None, logfile=None):
    '''
    :param pdb_ids: A set of PDB IDs to download
    :type pdb_ids: set(str)
    :param antibody_database_path: Path to the directory to save the fasta files to.
    :type antibody_database_path: str
    :param max_workers: Max number of workers in the thread pool while downloading.
    :type max_workers: int
    '''
    # download FASTA files
    fasta_files = [os.path.join(download_dir, pdb + '.fasta') for pdb in pdb_ids]
    download_url = 'https://www.rcsb.org/fasta/entry/{}'
    urls = [download_url.format(pdb) for pdb in pdb_ids]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = [
            executor.submit(lambda a: download_file(*a, sleep=sleep), args)
            for args in zip(urls, fasta_files)
        ]
        print('\ndownloading FASTA files:')
        sys.stdout.flush()
        for _ in tqdm(as_completed(results), total=len(urls)):
            pass
    sys.stdout.flush()
    downloaded_files = list_files(download_dir)
    
    # logging
    log_info = []
    n_downloads = len(downloaded_files)
    n_pdbs = len(pdb_ids)
    i = f"{n_downloads} of {n_pdbs} FASTAs were successfully downloaded"
    print(i)
    sys.stdout.flush()
    log_info.append(i)
    if n_pdbs > n_downloads:
        failed = []
        for pdb_id in pdb_ids:
            if not any([pdb_id in df for df in downloaded_files]):
                failed.append(pdb_id)
        log_info.append('The following PDB IDs were not downloaded:')
        log_info.extend(failed)
    
    if logfile is not None:
        write_log(logfile, '\n'.join(log_info))
        
        
def process_fastas(pdb_ids, 
                   raw_fasta_dir, 
                   processed_fasta_dir, 
                   pdb_dir,
                   logfile=None):
    '''
    
    '''
    # process FASTA files and record any failures
    warn_pdbs = []
    missing_pdbs = []
    print('\nprocessing fasta files:')
    sys.stdout.flush()
    for pdb_id in tqdm(pdb_ids):
        pdb_file = os.path.join(pdb_dir, f"{pdb_id}.pdb")
        if not os.path.isfile(pdb_file):
            missing_pdbs.append(pdb_id)
            continue
        raw_fasta_file = os.path.join(raw_fasta_dir, f"{pdb_id}.fasta")
        processed_fasta_file = os.path.join(processed_fasta_dir, f"{pdb_id}.fasta")
        warn = process_fasta(raw_fasta_file, 
                             processed_fasta_file,
                             pdb_file, 
                             pdb_id)
        if warn is not None:
            warn_pdbs.append(warn)

    # logging
    log_info = []
    if missing_pdbs:
        i = f'Processed PDB files were not found for {len(missing_pdbs)} PDB IDs'
        print(i)
        sys.stdout.flush()
        log_info.append(i + ':')
        log_info.append(', '.join(missing_pdbs))
    if warn_pdbs:
        w, s = ('were', 's') if len(warn_pdbs) > 1 else ('was', '')
        i = f"During processing, there {w} {len(warn_pdbs)} warning{s}"
        print(i)
        sys.stdout.flush()
        log_info.append(i + ':')
        log_info.append('\n'.join(warn_pdbs))
    else:
        log_info.append('All FASTA files were successfully processed')
    if logfile is not None:
        write_log(logfile, '\n'.join(log_info))
        

def process_fasta(raw_fasta_file,
                  processed_fasta_file,
                  pdb_file,
                  pdb_id):
    '''

    '''
    # read input FASTA file
    fasta_seqs = []
    for record in SeqIO.parse(raw_fasta_file, "fasta"):
        s = record.seq._data
        if type(s) == bytes:
            s = s.decode()
        fasta_seqs.append(s)
    if not fasta_seqs:
        err = f"NO FASTA SEQS: {pdb_id}\n"
        with open(raw_fasta_file, 'r') as f:
            err += f.read()
        return err

    # verify that both H and L chains are present
    try:
        pdb_h_seq = get_pdb_chain_seq(pdb_file, "H")
    except Exception as e:
        err = f'GET PDB H SEQ ERROR: {pdb_id}\n'
        err += traceback.format_exc()
        err += '\n'
        return err
    try:
        pdb_l_seq = get_pdb_chain_seq(pdb_file, "L")
    except Exception as e:
        err = f'GET PDB L SEQ ERROR: {pdb_id}\n'
        err += traceback.format_exc()
        err += '\n'
        return err
    if pdb_h_seq == None or len(pdb_h_seq) == 0:
        err = f'MISSING H CHAIN: {pdb_id}'
        return err
    if pdb_l_seq == None or len(pdb_l_seq) == 0:
        err = f'MISSING L CHAIN: {pdb_id}'
        return err

    # truncate the C-terminal end of H/L sequences
    cterm_length = 15
    trunc_h_seq = None
    h_cterm_seq = pdb_h_seq[-cterm_length:]
    for s in fasta_seqs:
        if h_cterm_seq in s:
            trunc_h_seq = s[:s.index(h_cterm_seq) + cterm_length]
    if trunc_h_seq is None:
        err = []
        err.append(f"\n\n{pdb_id}: H")
        alns = local_alignment(pdb_h_seq, targets=fasta_seqs)
        for aln in alns:
            err.append(aln.aligned_query)
            err.append(aln.alignment_midline)
            err.append(aln.aligned_target)
            return '\n'.join(err)
    trunc_l_seq = None
    l_cterm_seq = pdb_l_seq[-cterm_length:]
    for s in fasta_seqs:
        if l_cterm_seq in s:
            trunc_l_seq = s[:s.index(l_cterm_seq) + cterm_length]
    if trunc_l_seq is None:
        err = []
        err.append(f"\n\n{pdb_id}: L")
        alns = local_alignment(pdb_l_seq, targets=fasta_seqs)
        for aln in alns:
            err.append(aln.aligned_query)
            err.append(aln.alignment_midline)
            err.append(aln.aligned_target)
            return '\n'.join(err)

    # write the truncated sequences to file
    try:
        with open(processed_fasta_file, "w") as f:
            f.write(">{0}:H\n{1}\n>{0}:L\n{2}\n\n".format(
                pdb_id, trunc_h_seq, trunc_l_seq))
    except:
        err = []
        err.append("Failed to write truncated fasta for PDB ID {}".format(pdb_id))
        err.append("\tFasta file\t{}".format(raw_fasta_file))
        err.append("\tPDB file\t{}".format(pdb_file))
        return '\n'.join(err)



#---------------------------
#          utils
#---------------------------


def download_file(url, download_path, sleep=None):
    os.makedirs(os.path.dirname(download_path), exist_ok=True)
    if sleep is not None:
        time.sleep(sleep)
    with open(download_path, 'w') as f:
        f.write(requests.get(url).content.decode('utf-8'))


def write_log(log_file, log_data):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, 'w') as f:
        f.write(log_data)



#---------------------------
#          main
#---------------------------


def download_dataset(pdb_ids,
                     download_path,
                     sabdab_summary_file,
                     ignore_single_chains=True):
    """
    Downloads a training set from SAbDab, avoids downloading PDB files in the
    deepab/data/TestSetList.txt file.

    :param summary_file_path: Path to the summary file produced by SAbDab
    :type summary_file_path: str
    :param antibody_database_path: Path to the directory to save the PDB files to.
    :type antibody_database_path: str
    :param max_workers: Max number of workers in the thread pool while downloading.
    :type max_workers: int
    """
    # make required directories if they don't already exist
    raw_pdb_dir = os.path.join(download_path, 'raw_pdbs')
    processed_pdb_dir = os.path.join(download_path, 'pdbs')
    raw_fasta_dir = os.path.join(download_path, 'raw_fastas')
    processed_fasta_dir = os.path.join(download_path, 'fastas')
    log_dir = os.path.join(download_path, 'log')
    all_dirs = [raw_pdb_dir, processed_pdb_dir, raw_fasta_dir, processed_fasta_dir, log_dir]
    for d in all_dirs:
        os.makedirs(d, exist_ok=True)
    
    # download and process PDB files
    download_pdb_files(pdb_ids,
                       raw_pdb_dir,
                       logfile=os.path.join(log_dir, 'download_pdbs.txt'))
    process_pdbs(raw_pdb_dir,
                 processed_pdb_dir,
                 sabdab_summary_file,
                 ignore_single_chains,
                 logfile=os.path.join(log_dir, 'process_pdbs.txt'))
        

    # download and process FASTA files
    download_fasta_files(pdb_ids,
                         raw_fasta_dir, 
                         sleep=0.5, 
                         logfile=os.path.join(log_dir, 'download_fastas.txt'))
    process_fastas(pdb_ids,
                   raw_fasta_dir,
                   processed_fasta_dir,
                   processed_pdb_dir,
                   logfile=os.path.join(log_dir, 'process_fastas.txt'))


def run(input_file,
        download_path, 
        exclude_file=None, 
        sabdab_summary_file=None,
        ignore_single_chains=True,
        seqid=99,
        paired=True,
        nr_complex='All',
        nr_rfactor='',
        nr_res=4.0):
    '''
    
    '''
    # assemble the list of PDB IDs to download
    with open(input_file, 'r') as f:
        pdb_ids = [line.strip() for line in f]
    if exclude_file is not None:
        with open(exclude_file, 'r') as f:
            exclude = [line.strip() for line in f]
    else:
        exclude = []
    pdb_ids = [p for p in pdb_ids if p not in exclude]

    # download the SAbDab summary file if not provided
    if sabdab_summary_file is None:
        os.makedirs(download_path, exist_ok=True)
        sabdab_summary_file = os.path.join(download_path, 'sabdab_summary.tsv')
        download_sabdab_summary_file(sabdab_summary_file,
                                    seqid=seqid,
                                    paired=paired,
                                    nr_complex=nr_complex,
                                    nr_rfactor=nr_rfactor,
                                    nr_res=nr_res)

    # download and process the dataset
    download_dataset(pdb_ids,
                     download_path,
                     sabdab_summary_file,
                     ignore_single_chains=ignore_single_chains)



def _cli():
    desc = ('''
        Downloads chothia files from SAbDab
        ''')
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        help="Input file containing PDB IDs, one per line. Required.")
    parser.add_argument('-d', '--download_path', type=str,
                         help="Path to the download directory. If it does not exist, it will be created.")
    parser.add_argument('--exclude-file', type=str, default=None,
                        help="File containing PDB IDs to exclude from the download list. Useful if you have a test set that you want to ensure is not included when downloading a training set. Default is None.")
    parser.add_argument('--seqid', type=int, default=99,
                        help='Max sequence identity (%) allowed between two sequences downloaded from SAbDab. Default is 99')
    parser.add_argument('--allow-unpaired', dest='paired', default=True, action="store_false",
                        help='If set, allow downloading of unpaired Ab sequences from SAbDab. Default is False, which only downloads paired sequences.')
    parser.add_argument('--nr_complex', type=str, default='All', choices=['All', 'Bound only', 'Unbound only'],
                        help='In complex? Choices are: "All", "Bound only" or "Unbound only". Default is "All".')
    parser.add_argument('--nr_rfactor', type=str, default='',
                        help='R-Factor cutoff. Default is not to use a cutoff')
    parser.add_argument('--nr_res', type=int, default=4,
                        help='Resolution cutoff, in Angstroms. Default is 4')
    parser.add_argument('--sabdab-summary-file', default=None,
                        help='SAbDab summary file to use. If not provided, the summary file will be downloaded.')
    parser.add_argument('--allow-single-chains', default=True, action='store_false', dest='ignore_single_chains',
                        help='Allows H/L pairs that are on a single chain (scFv). Unlikely to work well, so use at your own risk.')
    args = parser.parse_args()

    run(args.input,
        args.download_path,
        exclude_file=args.exclude_file,
        sabdab_summary_file=args.sabdab_summary_file,
        ignore_single_chains=args.ignore_single_chains,
        seqid=args.seqid,
        paired=args.paired,
        nr_complex=args.nr_complex,
        nr_rfactor=args.nr_rfactor,
        nr_res=args.nr_res)



if __name__ == '__main__':
    _cli()
