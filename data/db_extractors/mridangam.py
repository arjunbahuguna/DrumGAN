import os
import argparse
import hashlib
import sys
import pickle
import requests

from utils.utils import get_date, mkdir_in_path, read_json, list_files_abs_path, get_filename, save_json
from random import shuffle
from tqdm import trange, tqdm

from .base_db import get_base_db, get_hash_dict

import ipdb
import numpy as np

__VERSION__ = "0.0.0"
MAX_N_FILES_IN_FOLDER = 10000

keys = ['instrument', 'audio-commons']
instruments = ['bheem', 'cha', 'dheem', 'dhin', 'num', 'ta', 'tha', 'thi', 'thom']
audio_commons_keys = ['duration', 'loudness', 'temporal_centroid', 'log_attack_time', 'hardness', 'depth', 'brightness', 'roughness', 'boominess', 'warmth', 'sharpness']     # Removed dynamic_range as it was 0 and was causing DivisionByZero error in "acval = (val[ac_att] - _min) / (_max - _min)"

keys.sort()
instruments.sort()
audio_commons_keys.sort()


def get_standard_format(path: str, dbname='mridangam'):
    """
    Populates attributes key in base_db dict to generate a standard format for any specified dataset

    Args:
        path (str): The path where the dataset is located.
        dbname (str, optional): The name of the dataset. Defaults to 'mridangam'.

    Returns:
        description (dict): A dictionary containing the standardized format for the dataset

    ToDo: Writes:
        filename.json: Write a json file for each wav file in dataset. Captures values from corresponding _analysis.json, generated using AudioCommons Audio Extractor. See 'out_item' dict.  todo ==> example of filename.json. todo ==> image of expected db folder structure
        dbname.json: Contains path to all filename.json files and other information. See 'description' dict. todo ==> image example of mridangam.json
    """

    description = get_base_db(dbname, __VERSION__)
    description_file = os.path.join(path, f'{dbname}.json')
    if os.path.exists(description_file):
        return read_json(description_file)
   
    root_dir = mkdir_in_path(path, f'{dbname}_standardized')
    extraction_config = os.path.join(root_dir, 'extraction_config.json')
    if os.path.exists(extraction_config):
        print("Extraction configuration exists. Loading...")
        return read_json(extraction_config)

    n_folders = 0
    attributes = {'instrument': {
                    'type': str(str),
                    'loss': 'xentropy', 
                    'values': instruments, 
                    'count': {i: 0 for i in instruments}}}
   
    i = 0
    for root, dirs, files in tqdm(os.walk(path)):
        if any(f.endswith('.wav') for f in files):
            inst_att = os.path.basename(root) #changed from dirname to basename
            
            files = list(filter(lambda x: x.endswith('.wav'), files)) #include wav files
            files = list(filter(lambda x: not x.startswith('._'), files))  #remove files with "._", these are usually metadata files on macOS
            files = list(filter(lambda x: os.path.exists(os.path.join(root, get_filename(x) + '_analysis.json')), files)) #include only those files which have an associated _analysis.json

            # Update the count
            attributes['instrument']['count'][inst_att] += len(files)

            # For every file, we will create a corresponding .json in the standard format
            # We will use out_item (line 89) to hold values of the .json, and keep appending to it
            for file in files:
                file = os.path.join(root, file)

                # Ensure number of files in a folder is less than 10k
                if i % MAX_N_FILES_IN_FOLDER == 0:
                    n_folders += 1
                    output_dir = mkdir_in_path(root_dir, f'folder_{n_folders}')
                
                filename = get_filename(file)

                # Add .json path to data
                output_file = os.path.join(output_dir, filename + '.json')
                description['data'].append(output_file)

                # Initialize out_items (henceforth called .json)
                out_item = {
                    'path': file,
                    'attributes': {}
                }
                
                # att is either instruments or audio-commons
                for att in keys:
                    if att == 'audio-commons':
                        if att not in attributes:
                            attributes[att] = {
                                'values': audio_commons_keys,
                                'type': str(float),
                                'loss': 'mse',
                                'max': {a: -1000.0 for a in audio_commons_keys},
                                'mean': {a: 0.0 for a in audio_commons_keys},
                                'min': {a: 1000.0 for a in audio_commons_keys},
                                'var': {a: 0.0 for a in audio_commons_keys}
                            }
                        ac_file = os.path.join(root, filename + '_analysis.json')
                        assert os.path.exists(ac_file), f"File {ac_file} does not exist"

                        #ac_atts: audio commons attributes for the current file
                        ac_atts = read_json(ac_file)
                        out_item['attributes'][att] = {}
                        
                        # ac_atts: _analysis.json  file values
                        # ac_att: audio_commons_keys values
                        for ac_att in audio_commons_keys:
                            if ac_att not in ac_atts: continue
                            acval = ac_atts[ac_att]

                            # For each file, copy _analysis.json values to .json 
                            out_item['attributes'][att][ac_att] = acval

                            # Capture max value for mridangam.json, the global dataset json
                            if acval > attributes[att]['max'][ac_att]:
                                attributes[att]['max'][ac_att] = acval

                            # Set min value for mridangam.json, the global dataset json
                            if acval < attributes[att]['min'][ac_att]:
                                attributes[att]['min'][ac_att] = acval

                            # Capture mean value for mridangam.json, the global dataset json
                            attributes[att]['mean'][ac_att] += acval
                    
                    if att not in attributes:
                        attributes[att] = {
                            'values': [],
                            'count': {}
                        }
                    
                    if att == 'instrument':
                        out_item['attributes'][att] = inst_att
                
                i+=1

                # Write .json for each file in dataset
                save_json(out_item, output_file)

    # Calculate mean for duration, loudness, dynamic range, etc
    for ac in audio_commons_keys:
       attributes['audio-commons']['mean'][ac] /= i
    
    description['attributes'] = attributes
    description['total_size'] = len(description['data'])
    
    # Write mridangam.json
    
    save_json(description, description_file)
    
    return description

def extract(path: str, criteria: dict={}, download: bool=False):
    """
    Extract and process data from a specified directory based on given criteria

    Args:
        path (str): Path to the directory containing the data to be extracted
        criteria (dict, optional): Criteria for filtering, balancing, normalizing
            Possible keys include:
            - 'balance': List of attributes to balance the extraction by
            - 'attributes': List of attributes to include in the extraction
            - 'size': Maximum size of the extracted data.
            - 'filter': Dictionary of attributes and their corresponding values to filter the data by
        download (bool, optional): Flag indicating whether to download additional data. Defaults to False

    Returns:
        tuple: A tuple containing:
            - data (list): List of file paths for the extracted data
            - metadata (list): List of metadata for the extracted data
            - extraction_dict (dict): A dictionary containing details about the extraction

    Writes:
        data.pt: todo
        extration.json: todo
         
    Notes:
        - This function first checks if the specified path exists. If it doesn't, it prints an error message and exits
        - It creates necessary directories for storing the extracted data
        - If data has already been extracted with the same criteria, it loads and returns the existing data
        - It processes the data based on the provided criteria, including filtering, balancing, and normalizing attributes
        - The function supports criteria for balancing the extracted data, filtering by specific attribute values, and limiting the size of the extraction
    """

    criteria_keys = ['balance', 'attributes', 'size', 'filter']
    criteria_keys.sort()

    if criteria != {}:
        assert all(k in criteria_keys for k in criteria),\
            "Filter criteria not understood"

    if not os.path.exists(path):
        print('Dataset folder not found')
        sys.exit(1)

    root_dir = mkdir_in_path(path, f'extractions')
    extraction_hash = get_hash_dict(criteria)

    extraction_dir = mkdir_in_path(root_dir, str(extraction_hash))
    data_pt_path = os.path.join(extraction_dir, 'data.pt')
    extraction_json_path = os.path.join(extraction_dir, 'extraction.json')
    if os.path.exists(data_pt_path):
        extraction_desc = read_json(extraction_json_path)

        print("Extraction.json exists!\n" \
              f"Loading {extraction_desc['name']}\n" \
              f"Version: {extraction_desc['version']}\n" \
              f"Date: {extraction_desc['date']}\n")
    
        return pickle.load(open(data_pt_path, 'rb'))
    
    standard_desc = get_standard_format(path)

    # Initialize important variables
    extraction_dict = get_base_db('mridangam', __VERSION__)
    attribute_list = list(standard_desc['attributes'].keys()) #['instrument', 'audio-commons']
    out_attributes = criteria.get('attributes', attribute_list) #out_attributes is criteria['attributes'] if it exists, otherwise it is attribute_list. created for filtering.
    out_attributes.sort()

    # Get dataset attribute values and counts 
    # Given the filtering criteria
    attribute_dict = {}
    for att in out_attributes: #If no criteria passed for attribute, out_attribute is ['instrument', 'audio-commons']

        # If att not present, define the attribute_dict format. This will be the same as attribute key in mridangam.json. See format: <attribute_dict.img link> todo
        if att not in attribute_dict:
            if att == 'audio-commons':
                attribute_dict[att] = {
                    'values': [],
                    'type': str(float),
                    'loss': 'mse',
                    'max': {},
                    'min': {},
                    'mean': {},
                    'var': {}
                }
            else: 
                attribute_dict[att] = {
                    'values': [],
                    'type': str(str),
                    'loss': 'xentropy',
                    'count': {}
                }

        # Is att present in criteria['filter']? if it is, then sort it
        if att in criteria.get('filter', {}).keys():
            criteria['filter'][att].sort()

            attribute_dict[att]['values'] = criteria['filter'][att]
            if att == 'audio-commons':
                for ac_att in criteria['filter'][att]:
                    attribute_dict[att]['max'][ac_att] = standard_desc['attributes'][att]['max'][ac_att]
                    attribute_dict[att]['min'][ac_att] = standard_desc['attributes'][att]['min'][ac_att]
                    attribute_dict[att]['mean'][ac_att] = standard_desc['attributes'][att]['mean'][ac_att]
                    attribute_dict[att]['var'][ac_att] = standard_desc['attributes'][att]['var'][ac_att] #changed value from 0.0 to var value from description dict
        else:
            attribute_dict[att] = standard_desc['attributes'][att].copy()

        attribute_dict[att]['values'].sort()
        attribute_dict[att]['count'] = {str(k): 0 for k in attribute_dict[att]['values']}

    data = []
    metadata = []
    size = criteria.get('size', standard_desc['total_size'])
    balance = False
    if 'balance' in criteria:
        balance = True
        b_atts = criteria['balance']

        for b_att in b_atts:
            count = []
            for v in attribute_dict[b_att]['values']:
                count.append(standard_desc['attributes'][b_att]['count'][str(v)])
            n_vals = len(count)
            size = min(size, n_vals * min(count))

    # Create progress bar for {todo}
    pbar = tqdm(standard_desc['data'])

    for file in pbar:
        item = read_json(file)

        item_atts = item['attributes']
        item_path = item['path']

        # Skip files that do not comply with
        # Filtered attribute criteria
        skip = False
        for att, val in item_atts.items():
            if att not in attribute_dict:
                continue
            if att == 'audio-commons':
                for ac_att in attribute_dict[att]['values']:
                    if ac_att not in val.keys():
                        skip = True
                        break
                    if np.isnan(val[ac_att]):
                        print(f"NaN value found in file {file} and att {ac_att}, skipping...")
                        skip = True
                        break
            elif att in criteria.get('filter', {}):
                if val not in criteria['filter'][att]:
                    skip = True
                    break
            else:
                if val not in attribute_dict[att]['values']: 
                    skip = True
                    break
        if skip: continue
        # Check balance of attributes
        if balance:
            for b_att in b_atts:
                val = item_atts[b_att]
                bsize = size / len(attribute_dict[b_att]['values'])
                if attribute_dict[b_att]['count'][str(val)] >= bsize:
                    skip = True
            if skip:
                continue
        # Store attribute index in list
        data_item = []
        for att in out_attributes:
            val = item_atts[att]
            # if attribute is multi-label (n out of m)
            if att == 'audio-commons':
                
                for ac_att in attribute_dict[att]['values']:
                    _max = attribute_dict['audio-commons']['max'][ac_att]
                    _min = attribute_dict['audio-commons']['min'][ac_att]                 
                    acval = (val[ac_att] - _min) / (_max - _min)
                    data_item += [acval]
                    attribute_dict[att]['var'][ac_att] += \
                        (attribute_dict[att]['mean'][ac_att] - val[ac_att])**2
            else:
                idx = attribute_dict[att]['values'].index(val)
                attribute_dict[att]['count'][str(val)] += 1
                data_item += [idx]
            # data_item.append(data_val)
        if skip: continue

        data.append(item_path)
        metadata.append(data_item)
        extraction_dict['data'].append(file)
        if len(data) >= size:
            pbar.close()
            break

    # compute std:
    
    if 'audio-commons' in attribute_dict:
        for att in attribute_dict['audio-commons']['values']:
            attribute_dict['audio-commons']['var'][att] /= len(data)

    extraction_dict['attributes'] = attribute_dict
    extraction_dict['output_file'] = data_pt_path
    extraction_dict['size'] = len(data)
    extraction_dict['hash'] = extraction_hash
    
    with open(data_pt_path, 'wb') as fp:
        pickle.dump((data, metadata, extraction_dict), fp)
    save_json(extraction_dict, extraction_json_path)
    return data, metadata, extraction_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='database extractor')
    parser.add_argument('db_path', type=str,
                         help='Path to the db root folder')
    
    parser.add_argument('-f', '--filter', help="Path to extraction configuration",
                        type=str, dest="filter_config", default=None)
    
    parser.add_argument('--download', action='store_true', 
                        help="Download db?",
                        dest="download", default=False)
    
    args = parser.parse_args()
    if args.filter_config != None:
        filter_config = read_json(args.filter_config)
    else:
        filter_config = {}

    # Removed dynamic_range as it was 0 and was causing DivisionByZero error in "acval = (val[ac_att] - _min) / (_max - _min)"
    filter_config = {
        'attributes': ['instrument', 'audio-commons'],
        'balance': [],
        'filter': {
            'instrument': ['kick', 'snare', 'cymbal'], # Include all instruments
            'audio-commons': [                          # Remove dynamic_range
                'duration', 'loudness', 'temporal_centroid', 'log_attack_time', 
                'hardness', 'depth', 'brightness', 'roughness', 'boominess', 
                'warmth', 'sharpness'
            ]
        }
    }
    extract(path=args.db_path,
            criteria=filter_config,
            download=args.download)    