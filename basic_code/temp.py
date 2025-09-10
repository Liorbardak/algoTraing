import os
import shutil
from pathlib import Path
inpath = "C:/Users/dadab/projects/algotrading/data/tickers"
outpath = "C:/Users/dadab/projects/algotrading/data/yahoo_tickers"

for subdir in [item.name for item in Path(inpath).iterdir() if item.is_dir()]:
    if os.path.isfile(os.path.join(inpath, subdir, 'StockPrice.csv')):
        os.makedirs(os.path.join(outpath, subdir), exist_ok=True)
        shutil.copy(os.path.join(inpath, subdir, 'StockPrice.csv') , os.path.join(outpath, subdir, 'StockPrice.csv'))