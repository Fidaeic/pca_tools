import os
from sklearn.preprocessing import StandardScaler
from pca_tools.model import PCA
import pandas as pd

# Ensure the references directory exists
references_dir = 'references'
if not os.path.exists(references_dir):
    os.makedirs(references_dir)

def create_preprocessed_numerical():
    # Setup
    data = pd.DataFrame({
        'numerical': [1, 2, 3, 4, 5],
        'non_numerical': [0, 1, 1, 0, 1]
    })
    data['numerical'] = StandardScaler().fit_transform(data[['numerical']])

    # Save the DataFrame as a Parquet file
    parquet_path = os.path.join(references_dir, 'preprocessed_numerical.parquet')
    data.to_parquet(parquet_path)

def create_preprocessed():
    # Setup
    data = pd.DataFrame({
        'numerical': [1, 2, 3, 4, 5],
        'non_numerical': [0, 1, 1, 0, 1]
    })
    data_trans = pd.DataFrame(StandardScaler().fit_transform(data), columns=data.columns, index=data.index)

    # Save the DataFrame as a Parquet file
    parquet_path = os.path.join(references_dir, 
                                'preprocessed_all.parquet')

    data_trans.to_parquet(parquet_path)

if __name__ == '__main__':

    create_preprocessed()
    create_preprocessed_numerical()