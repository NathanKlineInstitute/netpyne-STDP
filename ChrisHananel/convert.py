import os, sys, argparse
import pickle
sys.path.append(os.path.abspath(os.getcwd()))
sys.path.append(os.path.abspath(os.getcwd()) + '/neurosim/')
import netpyne
from sim import NeuroSim
from conf import read_conf, backup_config


def convert(config):
   
   ### ---Initialize--- ###
   dconf = read_conf(config)

   #### --- Set variabels--- ###
   sim_name = dconf['sim']['outdir'].split('/')[-1]
         
   # out_path uniquely identified per child
   global out_path
   out_path = os.path.join(os.getcwd(), 'results', f'{sim_name}')
     
   # Initialize the model with dconf config
   dconf['sim']['duration'] = 1e4
   dconf['sim']['recordWeightStepSize'] = 1e4
   dconf['sim']['outdir'] = out_path 
   model = NeuroSim(dconf, use_noise=False, save_on_control_c=False)
          
   ## loading
   with open(os.path.normpath(out_path + '/bestweights.pkl'), 'rb') as out:
      child_data = pickle.load(out)

   weights = child_data['best_weights']
   
    # set model weights 
   model.setWeightArray(netpyne.sim, weights)
   
   model.save()
   



if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("--config", type=str, default="None")
   
   args = parser.parse_args()
   
   if args.config =='None':
      print('forgot CONFIG! ')

   
   convert(**vars(args))