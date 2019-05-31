import pandas as pd
import numpy as np

def main():
    baseline = pd.read_csv('relational_ERM/data_cleaning/sbm_baseline_res.csv')
    table_output = baseline.groupby(['covariate', 'beta']).agg({"est_ate": [np.mean, np.std]})
    print(table_output)

if __name__ == '__main__':
    main()
