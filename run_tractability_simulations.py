import argparse
from torch_wrapper_objects import build_synthetic_datasets, run_training, LitM2M
from evaluation import calculate_rsquared
import pandas as pd

### running this file will store results in the `simulation_results` directory, 
### within a subdirectory that varies basaed on the simulation arguments

#### will be of the format `simulation_results/{args}/version_{trial number}/...
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


        
def run_analysis(case, seed=0, return_model=False):
    
    # from the case (as described in the overleaf), format the args for jen's code
    args = AttrDict({'N_met':case['J'], 
                     'N_bug':case['M'], 
                     'xdim':case['D'],
                     'ydim':case['D'], 
                     'ydi':case['D'],
		     'N_samples':case['N'], 
                     'K':case['K'], ### this is fed into inits
                     'L':case['L'], 
                     'lr':1e-2,#0.01,
                     'meas_var':0.1
                     })
    
    generated_successfully=False
    idx=0
    while not generated_successfully:
        try:
            ## build datasets
            train_dataset, val_dataset, gen_met_locs, gen_bug_locs = build_synthetic_datasets(args, 
                                                                                              generation_tp='linear',
                                                                                              seed=seed+idx*10)
            generated_successfully=True
        except:
            idx+=1
    
    # format args into a directory name
    case_path = '_'.join( [a.replace('_', '-') + '-' + str(b) 
                               for a,b in case.items()] ) 
    
    
    ## builds model, runs training, logs results
    fitted = run_training(train_dataset, 
                          val_dataset, 
                          gen_met_locs, 
                          gen_bug_locs, 
                          learning_rate=args.lr,
                          logger_path=case_path,
                          seed=seed
                         )
    
    train_r2, val_r2 = calculate_rsquared(fitted, train_dataset, val_dataset)
    
    if return_model:
        return(train_r2, val_r2, case_path, fitted)
    else:
        return(train_r2, val_r2, case_path)
    
def reload_data(checkpoint_path):
    case = {a[0]:int(a[1]) for a in [b.split('-') for b in checkpoint_path.split('/')[1].split('_')] }
    
    # from the case (as described in the overleaf), format the args for jen's code
    args = AttrDict({'N_met':case['J'], 
                     'N_bug':case['M'], 
                     'xdim':case['D'],
                     'ydim':case['D'], 
                     'ydi':case['D'],
                     'N_samples':case['N'], 
                     'K':case['K'], ### this is fed into inits
                     'L':case['L'], 
                     'lr':1e-2,#0.01,
                     'meas_var':0.1
                     })
    
    ## build datasets
    train_dataset, val_dataset, gen_met_locs, gen_bug_locs, \
                x, y, g, gen_beta, gen_alpha, gen_w, gen_z, gen_bug_locs, gen_met_locs, mu_bug, \
                    mu_met, r_bug, r_met = build_synthetic_datasets(args, return_all_info=True)
    
    return( x, y, g, gen_beta, gen_alpha, gen_w, gen_z, gen_bug_locs, gen_met_locs, mu_bug, \
                    mu_met, r_bug, r_met )
           
    
    

    
def main():
    # runs the simulations for all cases outlined in the overleaf scenarios
    
    # ordered lists of the different parametes for each case
    Ks = [2,2,2,3,10,10,10,10,20] # K: number of metabolite clusters
    Ls = [10]*5 # L: number of microbial clusters
    Ms=[200]*5 # M: number of microbial taxa
    Js=[800]*5 # J: number of metabolites
    Ns=[200]*5 # N: number of samples
    Ds=[10]*5 # D: embedding space
    
    case_summaries=[{'K':k,
                     'L':l,
                     'M':m,
                     'J':j,
                     'N':n,
                     'D':d } for k,l,m,j,n,d in zip(Ks, Ls, Ms, Js, Ns, Ds)]
    
    all_cases=[]
    all_train_r2s=[]
    all_val_r2s=[]
    
    for case in case_summaries:
        for seed in range(3):
            train_r2, val_r2, case_path = run_analysis(case, seed=seed)
            

            all_cases.append(case_path)
            all_train_r2s.append(train_r2)
            all_val_r2s.append(val_r2)
        
        pd.DataFrame({'Case':all_cases, 
                      'Train_r2':all_train_r2s, 
                      'Val_r2':all_val_r2s}).to_csv(\
                                'Tractability_Results/R2_summaries_varying_K.csv'
                                                   )
    

if __name__=='__main__':
    main()


 ### can ignore everyhint below here --- it's some commented out argparse code
    
    
    
    
    
    
    
    
 
    
    
    
    
    
    
# if __name__ == "__main__":
#     ## parse command line argmuents
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-lr", "--lr", help="learning rate", type=float, default = 0.01) #0.1
#     parser.add_argument("-fix", "--fix", help="params to fix", type=str, nargs='+', default = [])
#     parser.add_argument("-N_met", "--N_met", help="N_met", type=int, default = 10)
#     parser.add_argument("-N_bug", "--N_bug", help="N_bug", type=int, default = 10)
#     parser.add_argument("-L", "--L", help="number of microbe rules", type=int, default = 3)
#     parser.add_argument("-K", "--K", help="metab clusters", type=int, default = 5)
#     parser.add_argument("-meas_var", "--meas_var", help="measurment variance", type=float, default = 0.1)
#     parser.add_argument("-N_samples", "--N_samples", help="num of samples", type=int, default=1000)
#     parser.add_argument("-xdim", "--xdim", type=int, default=2, help = 'embedding dimension for microbes')
#     parser.add_argument("-ydim", "--ydim", type=int, default=2, help = 'embedding dimension for metabolites')
#     args = parser.parse_args()
    
    
#     run_analysis(args)
    
    
    
    
    


    
    
    
    
    
    
 #### GA: Below are the args I'm ignoring for the time being (focusing on the cases described in the overleaf)  
    
    

# parser.add_argument("-case", "--case", help="case", type=str,
#                     default = datetime.date.today().strftime('%m %d %Y').replace(' ','-'))

#     parser.add_argument("-iterations", "--iterations", help="number of iterations", type=int,default = 100)
#     parser.add_argument("-seed", "--seed", help = "seed for random start", type = int, default = 99)
#     parser.add_argument("-lb", "--lb", help = "whether or not to learn bug clusters", type = int, default = 0)
#     parser.add_argument("-lm", "--lm", help = "whether or not to learn metab clusters", type = int, default = 0)
#     parser.add_argument("-linear", "--linear", type = int, default = 1, help = 'whether to run linear model or not')
#     parser.add_argument("-nltype", "--nltype", type = str, default = "exp",
#                         help = 'if using synthetic data and linear == 0, how to non-linearly generate data'
#                                'choices are: exp, poly, sine, linear, sigmoid'
#                        )
#     parser.add_argument("-adjust_lr", "--adjust_lr", type=int, default=1,
#                         help = "whether to adjust the learning rate based on the size of the parameter")
#     parser.add_argument("-p_num", "--p_num", type=int, default=1,
#                         help = "if non-linear, how many neural networks per microbe cluster - metabolite cluster interaction"
#       
    
    
    
    #     parser.add_argument("-a_tau", "--a_tau", type=float, nargs = '+', default=[-0.01, -3], help = 'annealing for alpha temparature')
#     parser.add_argument("-w_tau", "--w_tau", type=float, nargs='+', default=[-0.01, -3], help = 'anealing for omega temperature')
#     parser.add_argument("-locs","--locs", type = str, default = 'true',
#                         help = 'true= use true microbe and metabolite embedded locations; '
#                                                                                'none= dont use locations;'
#                                                                                'rand= use random locations;')
#     parser.add_argument("-dtype", "--dtype", type=str, default='pubchem_tanimoto',
#                         help = "which type of distance embedding to use, choices are:" 
#                                " 'stratified', 'clumps', '', 'pubchem_tanimoto', 'RDK_tanimoto','MACCS_tanimoto' ")
#     parser.add_argument("-data", "--data", type = str, default = 'synthetic', help = "which input data to use; choices are: "
#                                                                                "'cdi', 'safari', 'synthetic' ")
#     parser.add_argument("-saf_type", "--saf_type", type=str, default='polar', help= 'if args.data == safari, which safari data type to run'
#                                                                                     'options are: polar, lipids-neg, or lipids-pos')

    
    
#     # Filtering criteria if args.data == 'cdi' or args.data == 'safari'
#     parser.add_argument("-nzm", "--non_zero_perc_met", type=float, default=80,
#                         help='percent of participants with non-zero metabolites in filtered data')
#     parser.add_argument("-nzb", "--non_zero_perc_bug", type=float, default=15,
#                         help='percent of participants with non-zero microbes in filtered data')
#     parser.add_argument("-cvm", "--coef_var_perc_met", type=float, default=5,
#                         help='coefficient of variation percentile for metabolites')
#     parser.add_argument("-cvb", "--coef_var_perc_bug", type=float, default=0,
#                         help='coefficient of variation percentile for microbes')

#     parser.add_argument("-most_corr", "--most_corr", type=int, default = 0,
#                         help = 'whether to use the data with high correlation bw microbes and metabolites or not')
