import argparse
from collections import Counter
from glob import glob
import h5py
import itertools
import json
import multiprocessing as mp
import os
import sys
from tqdm import tqdm

import numpy as np
import pandas as pd

from abutils.io import read_fasta, list_files, make_dir
from abutils.utils.jobs import monitor_mp_jobs
from abutils.utils.progbar import progress_bar

from anarci import anarci, number

import deepab
from deepab.util.util import _aa_dict, letter_to_num



#-------------------
#     utils
#-------------------


h_components = ["fwh1", "cdrh1", "fwh2", "cdrh2", "fwh3", "cdrh3", "fwh4"]
h_cdr_names = ["h1", "h2", "h3"]
l_components = ["fwl1", "cdrl1", "fwl2", "cdrl2", "fwl3", "cdrl3", "fwl4"]
l_cdr_names = ["l1", "l2", "l3"]

def to_dict(text):
    return json.loads(text.replace("\'", "\""))



#-------------------
#     ANARCI
#-------------------


def anarci_numbering(fasta_file, output_file=None, scheme='chothia', metadata_dict=None, chunksize=1000, verbose=True):
    '''
    ANARCI numbering
    '''
    # read sequences
    if verbose:
        print('reading input FASTA file...')
    seqs = read_fasta(fasta_file)
    if verbose:
        print('translating sequences...')
    translated_seqs = [(s.id, s.translate()) for s in seqs]
    seq_count = len(translated_seqs)
    if verbose:
        print(f"found {seq_count} sequences")
    
    # run ANARCI
    if verbose:
        print('\nrunning ANARCI:')
    p = mp.Pool()
    async_results = []
    for i in range(0, seq_count, chunksize):
        ts = translated_seqs[i:i+chunksize]
        async_results.append(p.apply_async(run_anarci, (ts, scheme)))
    monitor_mp_jobs(async_results, completion_string='\n')
    results = [ar.get() for ar in async_results]
    p.close()
    p.join()
    
    # process ANARCI output
    if verbose:
        print('processing ANARCI output...')
    numbering = []
    alignment_details = []
    hit_table = []
    for n, a, h in results:
        numbering.extend(n)
        alignment_details.extend(a)
        hit_table.extend(h)
        
    # build summary CSV
    if verbose:
        print('building summary CSV...')
    seq_nums = {'ANARCI_numbering_heavy': {},
                'ANARCI_numbering_light': {},
                'ANARCI_status_heavy': {},
                'ANARCI_status_light': {}}
    for num, aln, hit in zip(numbering, alignment_details, hit_table):
        d = {}
        if aln is None:
            continue
        aln = aln[0]
        chain = aln['chain_type']
        long_chain = 'heavy' if chain == 'H' else 'light'
        name = aln['query_name']
        for pos, res in num[0][0]:
            region = get_region(pos[0], chain)
            pos = f"{pos[0]}{pos[1].strip()}"
            if region not in d:
                d[region] = {}
            if res == '-':
                continue
            d[region][pos] = res
        seq_nums[f'ANARCI_numbering_{long_chain}'][name] = d
        seq_nums[f'ANARCI_status_{long_chain}'][name] = 'good'
    num_df = pd.DataFrame(seq_nums)
    if output_file is not None:
        metadata = str(metadata_dict) if metadata_dict is not None else '{}'
        csv = num_df.to_csv()
        with open(output_file, 'w') as f:
            f.write('\n'.join([metadata, csv]))
    else:
        return num_df
    
    
def run_anarci(seqs, scheme):
    return anarci(seqs, scheme=scheme, output=False)


def filter_anarci_csv(csv_file, print_progress=False, verbose=False):
    '''
    
    '''
    # read the metadata header
    rep_info = to_dict(
        np.genfromtxt(csv_file,
                      max_rows=1,
                      dtype=str,
                      delimiter="\t",
                      comments=None).item())
    info_dict = {
        "species": rep_info.get("Species", "None"),
        "isotype": rep_info.get("Isotype", "None"),
        "b_type": rep_info.get("BType", "None"),
        "b_source": rep_info.get("BSource", "None"),
        "disease": rep_info.get("Disease", "None"),
        "vaccine": rep_info.get("Vaccine", "None"),
    }

    # read the ANARCHI data
    col_names = pd.read_csv(csv_file, skiprows=1, nrows=1).columns
    oas_df = pd.read_csv(csv_file,
                         skiprows=1,
                         names=col_names,
                         header=None,
                         usecols=[
                             'ANARCI_status_light', 'ANARCI_status_heavy',
                             'ANARCI_numbering_heavy', 'ANARCI_numbering_light'
                         ])
    oas_df = oas_df.query(
        "ANARCI_status_light == 'good' and ANARCI_status_heavy == 'good'")
    oas_df = oas_df[['ANARCI_numbering_heavy', 'ANARCI_numbering_light']]

    # filter data with missing regions and build a data list
    data_list = []
    for index, (anarci_h_data, anarci_l_data) in enumerate(oas_df.values):
        anarci_h_data = to_dict(anarci_h_data)
        anarci_l_data = to_dict(anarci_l_data)
        missing_component = False
        for c in h_components:
            if not c in anarci_h_data:
                if verbose:
                    print(f"Missing heavy component in index {index}: {c}")
                missing_component = True
        for c in l_components:
            if not c in anarci_l_data:
                if verbose:
                    print(f"Missing heavy component in index {index}: {c}")
                missing_component = True
        if missing_component:
            continue

        h_prim, h_cdr_range_dict, h_cdr_seq_dict = extract_seq_components(
            anarci_h_data, h_components, h_cdr_names)
        l_prim, l_cdr_range_dict, l_cdr_seq_dict = extract_seq_components(
            anarci_l_data, l_components, l_cdr_names)

        data_list.append({
            "heavy_data": (h_prim, h_cdr_range_dict, h_cdr_seq_dict),
            "light_data": (l_prim, l_cdr_range_dict, l_cdr_seq_dict),
            "metadata":
            info_dict
        })

    return data_list


def combine_anarci_components(anarci_dict, components):
    seq_list = list(
        itertools.chain.from_iterable(
            [list(anarci_dict[c].values()) for c in components]))
    seq = "".join(seq_list)

    return seq



#-------------------
#     regions
#-------------------


def get_region(pos, chain):
    ends = region_ends[chain]
    c = 'h' if chain == 'H' else 'l'
    if pos < ends[f'fw{c}1']:
        return f'fw{c}1'
    elif pos < ends[f'cdr{c}1']:
        return f'cdr{c}1'
    elif pos < ends[f'fw{c}2']:
        return f'fw{c}2'
    elif pos < ends[f'cdr{c}2']:
        return f'cdr{c}2'
    elif pos < ends[f'fw{c}3']:
        return f'fw{c}3'
    elif pos < ends[f'cdr{c}3']:
        return f'cdr{c}3'
    else:
        return f'fw{c}4'


h_region_ends = {'fwh1': 26,
                 'cdrh1': 33,
                 'fwh2': 52,
                 'cdrh2': 57,
                 'fwh3': 96,
                 'cdrh3': 102}

k_region_ends = {'fwl1': 26,
                 'cdrl1': 33,
                 'fwl2': 50,
                 'cdrl2': 53,
                 'fwl3': 91,
                 'cdrl3': 97}

l_region_ends = {'fwl1': 26,
                 'cdrl1': 33,
                 'fwl2': 50,
                 'cdrl2': 53,
                 'fwl3': 91,
                 'cdrl3': 97}

region_ends = {'H': h_region_ends,
               'K': k_region_ends,
               'L': l_region_ends}



#-------------------
#    sequences
#-------------------


def extract_seq_components(anarci_dict, seq_components, cdr_names):
    seq = combine_anarci_components(anarci_dict, seq_components)

    cdr_range_dict = {}
    cdr_seq_dict = {}
    for i in range(0, 3):
        cdr_range_dict[cdr_names[i]] = [
            len(
                combine_anarci_components(anarci_dict,
                                          seq_components[:2 * i + 1])),
            len(
                combine_anarci_components(anarci_dict,
                                          seq_components[:2 * i + 2])) - 1
        ]
        cdr_seq_dict[cdr_names[i]] = combine_anarci_components(
            anarci_dict, ["cdr" + cdr_names[i]])

    return seq, cdr_range_dict, cdr_seq_dict


def sequences_to_h5(csv_dir,
                    output_file,
                    overwrite=False,
                    print_progress=False,
                    verbose=False):

    if verbose:
        print('filtering ANARCHI summary CSVs:')
    csv_files = glob(os.path.join(csv_dir, "*.csv"))
    data_list = []
    for csv in tqdm(csv_files):
        data_list.extend(
            filter_anarci_csv(csv, print_progress=False, verbose=verbose))

    num_seqs = len(data_list)
    max_h_len = 200
    max_l_len = 200

    if overwrite and os.path.isfile(output_file):
        os.remove(output_file)
    h5_out = h5py.File(output_file, 'w')
    h_len_set = h5_out.create_dataset('heavy_chain_seq_len', (num_seqs, ),
                                      compression='lzf',
                                      dtype='uint16',
                                      maxshape=(None, ),
                                      fillvalue=0)
    l_len_set = h5_out.create_dataset('light_chain_seq_len', (num_seqs, ),
                                      compression='lzf',
                                      dtype='uint16',
                                      maxshape=(None, ),
                                      fillvalue=0)
    h_prim_set = h5_out.create_dataset('heavy_chain_primary',
                                       (num_seqs, max_h_len),
                                       compression='lzf',
                                       dtype='uint8',
                                       maxshape=(None, max_h_len),
                                       fillvalue=-1)
    l_prim_set = h5_out.create_dataset('light_chain_primary',
                                       (num_seqs, max_l_len),
                                       compression='lzf',
                                       dtype='uint8',
                                       maxshape=(None, max_l_len),
                                       fillvalue=-1)
    h1_set = h5_out.create_dataset('h1_range', (num_seqs, 2),
                                   compression='lzf',
                                   dtype='uint16',
                                   fillvalue=-1)
    h2_set = h5_out.create_dataset('h2_range', (num_seqs, 2),
                                   compression='lzf',
                                   dtype='uint16',
                                   fillvalue=-1)
    h3_set = h5_out.create_dataset('h3_range', (num_seqs, 2),
                                   compression='lzf',
                                   dtype='uint16',
                                   fillvalue=-1)
    l1_set = h5_out.create_dataset('l1_range', (num_seqs, 2),
                                   compression='lzf',
                                   dtype='uint16',
                                   fillvalue=-1)
    l2_set = h5_out.create_dataset('l2_range', (num_seqs, 2),
                                   compression='lzf',
                                   dtype='uint16',
                                   fillvalue=-1)
    l3_set = h5_out.create_dataset('l3_range', (num_seqs, 2),
                                   compression='lzf',
                                   dtype='uint16',
                                   fillvalue=-1)
    species_set = h5_out.create_dataset('species', (num_seqs, ),
                                        compression='lzf',
                                        dtype=h5py.string_dtype())
    isotype_set = h5_out.create_dataset('isotype', (num_seqs, ),
                                        compression='lzf',
                                        dtype=h5py.string_dtype())
    b_type_set = h5_out.create_dataset('b_type', (num_seqs, ),
                                       compression='lzf',
                                       dtype=h5py.string_dtype())
    b_source_set = h5_out.create_dataset('b_source', (num_seqs, ),
                                         compression='lzf',
                                         dtype=h5py.string_dtype())
    disease_set = h5_out.create_dataset('disease', (num_seqs, ),
                                        compression='lzf',
                                        dtype=h5py.string_dtype())
    vaccine_set = h5_out.create_dataset('vaccine', (num_seqs, ),
                                        compression='lzf',
                                        dtype=h5py.string_dtype())

    if verbose:
        print('building h5-formatted output file:')
    for index, data_dict in tqdm(enumerate(data_list),
                                 disable=(not print_progress)):
        # Extract sequence from OAS data
        h_prim, h_cdr_range_dict, h_cdr_seq_dict = data_dict["heavy_data"]
        l_prim, l_cdr_range_dict, l_cdr_seq_dict = data_dict["light_data"]
        metadata = data_dict["metadata"]

        cdr_range_dict = {}
        cdr_range_dict.update(h_cdr_range_dict)
        cdr_range_dict.update(l_cdr_range_dict)

        h_len_set[index] = len(h_prim)
        l_len_set[index] = len(l_prim)

        h_prim_set[index, :len(h_prim)] = np.array(
            letter_to_num(h_prim, _aa_dict))
        l_prim_set[index, :len(l_prim)] = np.array(
            letter_to_num(l_prim, _aa_dict))

        for h_set, name in [(h1_set, 'h1'), (h2_set, 'h2'), (h3_set, 'h3'),
                            (l1_set, 'l1'), (l2_set, 'l2'), (l3_set, 'l3')]:
            h_set[index] = np.array(cdr_range_dict[name])

        species_set[index] = metadata["species"]
        isotype_set[index] = metadata["isotype"]
        b_type_set[index] = metadata["b_type"]
        b_source_set[index] = metadata["b_source"]
        disease_set[index] = metadata["disease"]
        vaccine_set[index] = metadata["vaccine"]



#-------------------
#      main
#-------------------


def run(input, output, verbose=True, overwrite=True):
    if os.path.isfile(input):
        fastas = [input, ]
    elif os.path.isdir(input):
        fastas = list_files(input)
    else:
        err = 'ERROR: input must be either a FASTA file or a directory containing FASTA files.'
        sys.exit(err)

    csv_dir = os.path.join(output, 'summary_csv')
    make_dir(csv_dir)

    for fasta in fastas:
        name = '.'.join(os.path.basename(fasta).split('.')[:-1])
        dashes = '=' * (len(name) + 4)
        if verbose:
            print('  ' + name)
            print(dashes)
        csv_file = os.path.join(csv_dir, f"{name}.csv")
        anarci_numbering(fasta, csv_file)
        if verbose:
            print('\n')

    sequences_to_h5(csv_dir, os.path.join(output, 'PairedSeq.h5'), overwrite=overwrite, verbose=verbose)


def cli():
    desc = 'Creates an h5-formatted PairedSeq file from one or more FASTA files. \
            PairedSeq h5 files can be used to train the DeepAb BiLSTM.'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Input, either a single FASTA-formatted file or a directory containing \
        FASTA-formatted files. Required.'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Directory for output files, including the h5 file and ANARCI summary CSVs. \
        If the directory does not exist, it will be created. Required.'
    )
    parser.add_argument(
        '--quiet',
        dest='verbose',
        default=True,
        action='store_false',
        help='If set, progress reporting will be silenced. Default is to print progess.'
    )
    parser.add_argument(
        '--overwrite',
        action="store_true",
        help='If set, PairedSeq h5 files will be overwritten if they already exist. \
        Default is False.',
        default=False
    )
    args = parser.parse_args()
    run(args.input, args.output, args.verbose, args.overwrite)



if __name__ == '__main__':
    cli()
