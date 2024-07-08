import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pickle_utils import load_post_process
import os
import seaborn as sns
from scipy import stats
import seaborn as sns
from scipy import stats
import argparse
import json
import logging
import os
import sys
from typing import Dict, Any
import pandas as pd
from pickle_utils import load_post_process
from np_analysis import generate_contour_plots, analyze_data, correlation_matrix, trend_analysis, find_non_constant_variables


def setup_logging(log_level: str) -> None:
    """Set up logging configuration."""
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""

    with open(config_file, 'r') as f:
        return json.load(f)

def main(args: argparse.Namespace) -> None:
    """Main execution function."""
    config = load_config(args.config)
    if not config['modes_solved']:
        print(f"Finite DFB mode solutions have not been calculated - please run DFBsolve_NP.py")
    setup_logging(args.log_level)

    input_file = config['results_fn']
    logging.info(f"Loading data from {input_file}")
    sr, ar, param, inputs = load_post_process(input_file)

    csv_filename = f'{os.path.splitext(input_file)[0]}_Results.csv'
    analyze_data(sr, ar, param, inputs, csv_filename)

    all_data = pd.concat([sr, param, inputs], axis=1)
    correlation_variables = find_non_constant_variables(all_data)

    logging.info("Non-constant variables detected:")
    logging.info(correlation_variables)

    if args.generate_plots:
        logging.info(f"Generating contour plots for {args.plot_variable}")
        generate_contour_plots(sr, param, config['COL_1'], config['COL_2'], args.plot_variable)

    if args.correlation_matrix:
        logging.info("Generating correlation matrix")
        correlation_matrix(all_data, correlation_variables, 
                           save_path=os.path.join(args.output_dir, 'correlation_matrix.png'))

    if args.trend_analysis:
        logging.info("Performing trend analysis")
        trend_analysis(sr, param)

    logging.info("Analysis complete. Plots have been generated and saved.")


# Default arg:
# python Analysis.py --config Data/EE_NP_JSON_template.json --output-dir Data/plots --log-level INFO --generate-plots --plot-variable Jth --correlation-matrix --trend-analysis

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze DFB laser data.")
    parser.add_argument('--config', default='EE_DFB_comparing_with_text.json', help="Path to configuration file")
    parser.add_argument('--output-dir', default='output', help="Directory to save output files")
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    parser.add_argument('--generate-plots', action='store_true', help="Generate contour plots")
    parser.add_argument('--plot-variable', default='Jth', help="Variable to plot in contour plots")
    parser.add_argument('--correlation-matrix', action='store_true', help="Generate correlation matrix")
    parser.add_argument('--trend-analysis', action='store_true', help="Perform trend analysis")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
    
    