import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

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

def create_scores():
    # Generate random data
    np.random.seed(42)  # For reproducibility
    data = np.random.rand(100, 5)  # 100 samples, 5 features
    columns = [f'feature_{i}' for i in range(1, 6)]

    data = StandardScaler().fit_transform(data)

    pca = PCA(n_components=3, )

    scores = pca.fit_transform(data)

    pc_columns = [f'PC_{i}' for i in range(1, 4)]

    dataframe = pd.DataFrame(scores, columns=pc_columns)

    parquet_path = os.path.join(references_dir, 'scores.parquet')
    dataframe.to_parquet(parquet_path)


if __name__ == '__main__':

    create_preprocessed()
    create_preprocessed_numerical()
    create_scores()