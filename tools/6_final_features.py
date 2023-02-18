import pandas as pd
from pathlib import Path
if __name__ == '__main__':
    FILE = '/data1/riverside/blur_dataset_features/features.parquet'
    path = Path(FILE)
    df = pd.read_parquet(FILE)
    df_group = df.groupby(['img_num']).mean()
    df_final = df_group.dropna()
    df_final.to_csv(path.parent/'features_final.csv', index=None, header=True)
