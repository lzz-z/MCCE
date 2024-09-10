import argparse
from model.MOLLM import MOLLM  # Ensure that MOLLM class is correctly imported from its respective module

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run MOLLM with a configuration file')
    
    # Add an argument for the configuration file path
    parser.add_argument(
        '--config', type=str, required=True, help='Path to the configuration file (YAML format)'
    )
    
    # Parse the arguments from the command line
    args = parser.parse_args()

    # Pass the config file path to MOLLM and run it
    mollm = MOLLM(args.config)
    mollm.run()

if __name__ == "__main__":
    main()
