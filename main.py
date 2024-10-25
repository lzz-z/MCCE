import argparse
from model.MOLLM import MOLLM  # Ensure that MOLLM class is correctly imported from its respective module

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run MOLLM with a configuration file')
    
    # Add an argument for the configuration file path
    parser.add_argument('config', type=str, default='/home/v-nianran/src/MOLLM/config/chem/filter_logs_red_sa_sim.yaml',  help='Path to the configuration file (YAML format)')
    parser.add_argument('--resume', action='store_true', help='resume training from the last checkpoint')
    parser.add_argument('--eval', action='store_true', help='evaluate this results according to the yaml file')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    
    # Parse the arguments from the command line
    args = parser.parse_args()

    # Pass the config file path to MOLLM and run it
    print('resume:',args.resume)
    #args.eval = True
    mollm = MOLLM(args.config,resume=args.resume,eval=args.eval,seed=args.seed)
    
    if args.eval:
        print(f'start evaluation of {args.config}')
        mollm.load_evaluate()
    else:
        mollm.run()

if __name__ == "__main__":
    main()
