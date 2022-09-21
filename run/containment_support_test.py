import os
import sys
from tqdm import tqdm, trange

sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

from simple_relational_reasoning.embeddings.containment_support_dataset import ContainmentSupportDataset

if __name__ == '__main__':
    dataset = ContainmentSupportDataset('/Users/guydavidson/projects/BlockWorld/outputs/containtment_all_objects')
    print(dataset.dataset.shape)
    print(dataset.target_objects)
    print(dataset.n_configurations)
    print(dataset.index_zfill)
    print(dataset.scene_types)
       