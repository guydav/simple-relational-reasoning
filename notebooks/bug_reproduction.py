# First Cell
import os
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../run'))

from collections import defaultdict
import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
import typing
import pandas as pd
# Only change I made
# from tqdm.notebook import tqdm
from tqdm import tqdm

from torchvision.transforms import functional as F

from IPython.display import display, Markdown

from simple_relational_reasoning.embeddings.stimuli import build_differet_shapes_stimulus_generator, build_split_text_stimulus_generator, build_random_color_stimulus_generator,\
    find_non_empty_indices, EMPTY_TENSOR_PIXEL
from simple_relational_reasoning.embeddings.triplets import QuinnTripletGenerator, ABOVE_BELOW_RELATION, BETWEEN_RELATION
from simple_relational_reasoning.embeddings.visualizations import filter_and_group, DEFAULT_TEXT_KWARGS, save_plot

# Second cell
BASELINE_TRIPLET_KWARGS = dict(n_target_types=2, extra_diagonal_margin=0)

def parse_above_below_condition(df):
    above_below_types = []

    for _, (relation, two_refs, adjacent_refs) in \
        df.loc[:, ['relation', 'two_reference_objects', 'adjacent_reference_objects']].iterrows():
        a_b_type = None

        if relation == 'above_below':
            if not two_refs:
                a_b_type = 'one_reference'

            elif adjacent_refs:
                a_b_type = 'adjacent_references'

            else:
                a_b_type = 'gapped_references'

        above_below_types.append(a_b_type)

    return df.assign(above_below_type=above_below_types)


GENERATOR_NAMES = ('color bar', 'split text', 'random colors')


def create_stimulus_generators_and_names(names=GENERATOR_NAMES, seed=None, **kwargs):
    rng = np.random.default_rng(seed if seed is not None else np.random.randint(0, 2**32))

    generators = (
        build_differet_shapes_stimulus_generator(rng=rng, **kwargs), 
        
        build_split_text_stimulus_generator(
            # reference_box_size=10,
            # total_reference_size=(10, 140), n_reference_patches=8,
            # reference_patch_kwargs=dict(ylim=(-70, 70)),
            rng=rng, **kwargs),
        build_random_color_stimulus_generator(rng=rng, **kwargs)
    )
    
    return zip(generators, names)

# Third cell
DATA_PATH = 'centroid_sizes.csv'
LIST_COLUMNS = [
    'row_centroids', 'col_centroids', 
    'first_non_empty_row', 'last_non_empty_row', 
    'first_non_empty_col', 'last_non_empty_col'
]
COLUMNS = [
    'relation', 'two_reference_objects', 
    'adjacent_reference_objects', 'transpose', 
    'n_habituation_stimuli', 'rotate_angle', 'stimulus_generator'
] + LIST_COLUMNS
N_examples = 100
ANGLES = [0, 30, 45, 60, 90, 120, 135, 150]

OPTION_SET = (
    (ABOVE_BELOW_RELATION, BETWEEN_RELATION),
    (False, True),
    (False, True),
    (False, True),
    [1, 4],
    ANGLES
) 

DISTANCE_ENDPOINTS_DICT = {
    (ABOVE_BELOW_RELATION, False, False): (30, 80),
    (ABOVE_BELOW_RELATION, True, False): (60, 115),
    (ABOVE_BELOW_RELATION, True, True): (40, 80),
    (BETWEEN_RELATION, True, False): (50, 80),
}


# Fourth cell
generate_data = True
if generate_data:
    total_options = np.prod([len(v) for v in OPTION_SET])
    option_iter = itertools.product(*OPTION_SET)

    data_rows = []

    for relation, two_reference_objects, adjacent_reference_objects, transpose, n_habituation_stimuli, rotate_angle in tqdm(option_iter, total=total_options):
        if (relation == ABOVE_BELOW_RELATION) and not two_reference_objects and adjacent_reference_objects:
            continue

        if (relation == BETWEEN_RELATION) and ((not two_reference_objects) or adjacent_reference_objects):
            continue

        distance_endpoints = DISTANCE_ENDPOINTS_DICT[(relation, two_reference_objects, adjacent_reference_objects)]

        print(f'relation={relation} two_refs={two_reference_objects} adj_refs={adjacent_reference_objects} transpose={transpose} n_hs={n_habituation_stimuli} angle={rotate_angle} endpoints={distance_endpoints}')

        for stimulus_generator, generator_name in create_stimulus_generators_and_names(rotate_angle=rotate_angle):
            triplet_generator = QuinnTripletGenerator(stimulus_generator, distance_endpoints,
                relation=relation, two_reference_objects=two_reference_objects,
                adjacent_reference_objects=adjacent_reference_objects, 
                transpose=transpose,
                n_habituation_stimuli=n_habituation_stimuli,
                track_centroids=True,
                **BASELINE_TRIPLET_KWARGS)

            triplets = triplet_generator(N_examples, normalize=False)
            non_empty_tuples = [find_non_empty_indices(t, empty_value=EMPTY_TENSOR_PIXEL, color_axis=0) for t in triplets.view(-1, *triplets.shape[2:])]
            del triplets
            row_centroids, col_centroids = zip(*triplet_generator.stimulus_centroids)
            row_centroids = list(row_centroids)
            col_centroids = list(col_centroids)

            first_non_empty_row, last_non_empty_row, first_non_empty_col, last_non_empty_col = zip(*non_empty_tuples)
            first_non_empty_row = [i.item() for i in first_non_empty_row]
            last_non_empty_row = [i.item() for i in last_non_empty_row]
            first_non_empty_col = [i.item() for i in first_non_empty_col]
            last_non_empty_col = [i.item() for i in last_non_empty_col]
            
            row = [relation, two_reference_objects, adjacent_reference_objects, transpose,
                n_habituation_stimuli, rotate_angle, generator_name, 
                row_centroids, col_centroids, 
                first_non_empty_row, last_non_empty_row, first_non_empty_col, last_non_empty_col
            ]
            data_rows.append(row)

            del triplet_generator
            del stimulus_generator

    
    # data_df = pd.DataFrame(data_rows, columns=COLUMNS)
    # data_df = parse_above_below_condition(data_df)
    # data_df.to_csv(DATA_PATH, index=False)

else:
    data_df = pd.read_csv(DATA_PATH, converters={col: pd.eval for col in LIST_COLUMNS})    
    data_df = data_df.assign(**{col: data_df[col].apply(list) for col in LIST_COLUMNS})
    data_df = data_df.assign(relation_and_type=data_df.above_below_type)
    data_df.relation_and_type[data_df.relation_and_type.isna()] = 'between'
    

# data_df.head() 

