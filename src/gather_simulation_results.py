#!/usr/bin/env python

import pandas as pd
import os
import numpy as np



inp = pd.read_csv('20250221_simu2025.csv', header=None, names=['idx', 'folderpath'])
tables = list()
non_working = list()
for index, row in list(inp.iterrows()):
    filename = '{}/estimation_results_complete.csv'.format(row.folderpath)
    try:
        timestamp = os.path.getmtime(filename)
        df = pd.read_csv(filename, sep='\t')
        df = df.assign(timestamp=timestamp)
        tables.append(df)
    except FileNotFoundError:
        print(index, filename)
        non_working.append(index + 1)


big_table2 = pd.concat(tables, axis=0)
big_table2.to_csv('results/20250313_big_table_bootstrap.csv', sep='\t', index=False)