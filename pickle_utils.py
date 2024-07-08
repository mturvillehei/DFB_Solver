import pickle
import pandas as pd
import os
def write_pickle(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved the sorted COMSOL data to {filename} in {os.getcwd()}")    

def read_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def add_metadata(filename, metadata):
    data = read_pickle(filename)
    if 'metadata' not in data:
        data['metadata'] = {}
    data['metadata'].update(metadata)
    write_pickle(filename, data)

def save_sort(output_fn, param_columns, final_results):
    data = {
        'parameters': {col: final_results[col].tolist() for col in param_columns},
        'inputs': {col: final_results[col].tolist() for col in final_results.columns if col not in param_columns}
    }
    write_pickle(output_fn, data)

def save_post_process(fn, scalar_results, array_results):
    data = read_pickle(fn)
    
    # Update with new results, preserving all existing keys
    data.update({
        'scalar_results': scalar_results.to_dict(),
        'array_results': array_results
    })
    
    write_pickle(fn, data)
    
def load_post_process(fn):
    data = read_pickle(fn)
    scalar_results = pd.DataFrame(data['scalar_results'])
    array_results = data['array_results']
    parameters = pd.DataFrame(data['parameters'])
    inputs = pd.DataFrame(data['inputs'])
    return scalar_results, array_results, parameters, inputs