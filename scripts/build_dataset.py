import os
import json
import numpy as np
#sys.path.append('..\\AMN_refactor\\amn')
os.chdir("c:\\Users\\tnhoang\\Documents\\refactor_AMN\\AMN_refactor")

from amn.dataset import SimulatedDataset
from amn.dataset import ExperimentalDataset
from amn.dataset import MetabolicDataset


def generate_dataset(dataset_name,generation_parameters_file,data_dir,verbose=False):
    
    with open(generation_parameters_file, 'r') as f:
        data = json.load(f)

    params = data[dataset_name]
    
    dataset_dict = dict(params["MetabolicDataset_parameters"])
    dataset_dict["cobra_file"] = data_dir + params["MetabolicDataset_parameters"]["cobra_file"]
    dataset_dict["medium_file"] = data_dir + params["MetabolicDataset_parameters"]["medium_file"]

    if "experimental_file" in  params["MetabolicDataset_parameters"]:
        dataset_dict["experimental_file"] = data_dir + params["MetabolicDataset_parameters"]["experimental_file"]

    if params["method_generation"] == "simulated":
        dataset = SimulatedDataset(**dataset_dict)
    elif params["method_generation"] == "experimental":
        dataset = ExperimentalDataset(**dataset_dict)
    
    saving_params = dict(params["saving_parameters"])
    saving_params["dataset_dir"] = data_dir + params["saving_parameters"]["dataset_dir"]
    
    dataset.save(**saving_params)

    if verbose : 
        dataset = MetabolicDataset(dataset_file=os.path.join(saving_params["dataset_dir"],
                                                             saving_params["dataset_name"]+'.npz'))
        dataset.printout()


if __name__ == "__main__":

    seed = 10
    generation_file = "config/dataset_generation.json"
    data_dir = "data"

    dataset_name = "iML1515_EXP_UB"
    dataset_name = "biolog_iML1515_medium_UB"
    dataset_name = "e_coli_core_UB_100"
    dataset_name = "IJN1463_10_UB"
    dataset_name = "iML1515_UB"
    dataset_name = "IJN1463_EXP_UB"
    dataset_name = "iML1515_EXP_2_UB"
    dataset_name = "iML1515_Paul_280"
    dataset_name = "iML1515_Paul_EXP"

    np.random.seed(seed=10)  
    generate_dataset(dataset_name,generation_file,data_dir,verbose=False)

