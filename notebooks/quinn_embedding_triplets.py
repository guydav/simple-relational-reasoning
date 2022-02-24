import numpy as np
import torch
import tqdm

from abc import abstractmethod
from functools import lru_cache
from tqdm.notebook import tqdm


DEFAULT_RANDOM_SEED = 33
TRIPLET_CACHE_SIZE = 16

class TripletGenerator:
    def __init__(self, stimulus_generator, 
                 two_reference_objects=False, two_targets_between=True, n_target_types=1,
                 transpose=False, vertical_margin=0, horizontal_margin=0, seed=DEFAULT_RANDOM_SEED, use_tqdm=False):
        self.stimulus_generator = stimulus_generator
        self.two_reference_objects = two_reference_objects
        self.two_targets_between = two_targets_between
        self.n_target_types = n_target_types
        
        if n_target_types > self.stimulus_generator.n_target_types:
            raise ValueError(f'Expected n_target_types={n_target_types} <= self.stimulus_generator.n_target_types={self.stimulus_generator.n_target_types}')
        
        self.transpose = transpose
        self.vertical_margin = vertical_margin
        self.horizontal_margin = horizontal_margin
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.use_tqdm = use_tqdm
        
    @lru_cache(maxsize=TRIPLET_CACHE_SIZE)
    def __call__(self, n=1, normalize=True, seed=None):
        if seed is None:
            seed = self.seed
            
        self.rng = np.random.default_rng(self.seed)
        n_iter = range(n)
        if self.use_tqdm:
            n_iter = tqdm(range(n), desc='Data Generation')
        
        results = [self.generate_single_triplet(normalize=normalize)for _ in n_iter]   
        result_tensor = torch.stack(results)
        
        if self.transpose:
            return result_tensor.permute(0, 1, 2, 4, 3)
        
        return result_tensor
    
    @abstractmethod
    def generate_single_triplet(self, normalize=True):
        pass
    

class EquilateralTripletGenerator(TripletGenerator):
    def __init__(self, stimulus_generator, side_length_endpoints, 
                 target_margin_from_reference_edge=0, pair_above=None,
                 two_reference_objects=False, two_targets_between=True, 
                 n_target_types=1, transpose=False, 
                 vertical_margin=0, horizontal_margin=0, 
                 seed=DEFAULT_RANDOM_SEED, use_tqdm=False):
        super(EquilateralTripletGenerator, self).__init__(
            stimulus_generator=stimulus_generator, 
            two_reference_objects=two_reference_objects,
            two_targets_between=two_targets_between, n_target_types=n_target_types,
            transpose=transpose, vertical_margin=vertical_margin,
            horizontal_margin=horizontal_margin, seed=seed, use_tqdm=use_tqdm)
        
        if not hasattr(side_length_endpoints, '__len__'):
            side_length_endpoints = (side_length_endpoints, side_length_endpoints)
            
        self.side_length_endpoints = side_length_endpoints
        self.target_margin_from_reference_edge = target_margin_from_reference_edge
        self.pair_above = pair_above
        
    def generate_single_triplet(self, normalize=True):
        side_length = self.rng.integers(*self.side_length_endpoints)
        height = (3 ** 0.5) * side_length / 2
        half_height = height // 2
        if self.two_reference_objects:
            min_vertical_margin = height + (self.stimulus_generator.reference_size[0] // 2) + 1
        else:
            min_vertical_margin = half_height + (self.stimulus_generator.target_size[0] // 2) + 1

        vertical_margin = max(min_vertical_margin, self.vertical_margin)
        horizontal_margin = max(self.stimulus_generator.reference_size[1] // 2 + 1, self.horizontal_margin)

        reference_center_position = np.array(
            (self.rng.integers(vertical_margin, self.stimulus_generator.canvas_size[0] - vertical_margin),
             self.rng.integers(horizontal_margin, self.stimulus_generator.canvas_size[1] - horizontal_margin)), 
            dtype=np.int)

        target_margin = (self.stimulus_generator.reference_size[1] - self.stimulus_generator.target_size[1]) // 2
        left_target_horizontal_offset = self.rng.integers(-target_margin, target_margin - side_length)
        middle_target_horizontal_offset = left_target_horizontal_offset + side_length // 2
        right_target_horizontal_offset = left_target_horizontal_offset + side_length

        if self.pair_above is None:
            pair_above = np.sign(self.rng.uniform(-0.5, 0.5))
        else:
            pair_above = self.pair_above and 1 or -1

        left_target_offset = np.array((pair_above * half_height, left_target_horizontal_offset), dtype=np.int)
        middle_target_offset = np.array((-1 * pair_above * half_height, middle_target_horizontal_offset), dtype=np.int)
        right_target_offset = np.array((pair_above * half_height, right_target_horizontal_offset), dtype=np.int)

        target_positions = [tuple(reference_center_position + offset) for offset in 
                            (left_target_offset, right_target_offset, middle_target_offset)]

        if self.two_reference_objects:
            second_reference_center = np.copy(reference_center_position)
            two_targets_between = self.two_targets_between and 1 or -1
            second_reference_vertical_direction = pair_above * two_targets_between
            second_reference_center[0] += second_reference_vertical_direction * height
            reference_center_position = [reference_center_position, second_reference_center]
            
        if self.n_target_types == 1:
            target_indices = (self.rng.integers(0, self.stimulus_generator.n_target_types), ) * 3
        if self.n_target_types == 2:
            pair_color = self.rng.uniform() > 0.5
            single_color = 1 - pair_color
            if self.rng.uniform() > 0.5:
                target_indices = (pair_color, single_color, pair_color)
            else:
                target_indices = (single_color, pair_color, pair_color)
        elif self.n_target_types == 3:
            target_indices = [0, 1, 2]
            self.rng.shuffle(target_indices)
            target_indices = tuple(target_indices)
            
#         if second_reference_center[0] <= (self.stimulus_generator.reference_size[0] // 2) or \
#             second_reference_center[0] >= (223 - (self.stimulus_generator.reference_size[0] // 2)) or \
#             second_reference_center[1] <= (self.stimulus_generator.reference_size[1] // 2) or \
#             second_reference_center[0] >= (223 - (self.stimulus_generator.reference_size[0] // 2)):
            
#             print(reference_center_position)
#             print(side_length, height, half_height, horizontal_margin, vertical_margin)
        
        return self.stimulus_generator.batch_generate(target_positions, 
                                                      reference_center_position, 
                                                      target_indices, 
                                                      normalize=normalize,
                                                      transpose_target=self.transpose)
    
    
class QuinnTripletGenerator(TripletGenerator):
    def __init__(self, stimulus_generator, distance_endpoints, 
                 target_margin_from_reference_edge=0, pair_above=None, two_objects_left=None,
                 two_reference_objects=False, two_targets_between=True, 
                 n_target_types=1, transpose=False,
                 vertical_margin=0, horizontal_margin=0,
                 seed=DEFAULT_RANDOM_SEED, use_tqdm=False):
        super(QuinnTripletGenerator, self).__init__(
            stimulus_generator=stimulus_generator, 
            two_reference_objects=two_reference_objects,
            two_targets_between=two_targets_between, n_target_types=n_target_types,
            transpose=transpose, vertical_margin=vertical_margin, 
            horizontal_margin=horizontal_margin, seed=seed, use_tqdm=use_tqdm)
        
        if not hasattr(distance_endpoints, '__len__'):
            distance_endpoints = (distance_endpoints, distance_endpoints)
            
        self.distance_endpoints = distance_endpoints        
        self.target_margin_from_reference_edge = target_margin_from_reference_edge
        self.pair_above = pair_above
        self.two_objects_left = two_objects_left
    
    def generate_single_triplet(self, normalize=True):
        distance = self.rng.integers(*self.distance_endpoints)
        half_distance = distance // 2
        if self.two_reference_objects:
            min_vertical_margin = distance + (self.stimulus_generator.reference_size[0] // 2) + 1
        else:
            min_vertical_margin = half_distance + (self.stimulus_generator.target_size[0] // 2) + 1

        vertical_margin = max(min_vertical_margin, self.vertical_margin)
        horizontal_margin = max(self.stimulus_generator.reference_size[1] // 2 + 1, self.horizontal_margin)

        reference_center_position = np.array(
            (self.rng.integers(vertical_margin, self.stimulus_generator.canvas_size[0] - vertical_margin),
             self.rng.integers(horizontal_margin, self.stimulus_generator.canvas_size[1] - horizontal_margin)), 
            dtype=np.int)

        target_margin = (self.stimulus_generator.reference_size[1] - self.stimulus_generator.target_size[1]) // 2
        left_target_horizontal_offset = self.rng.integers(-target_margin, target_margin - distance)
        right_target_horizontal_offset = left_target_horizontal_offset + distance

        if self.pair_above is None:
            pair_above = np.sign(self.rng.uniform(-0.5, 0.5))
        else:
            pair_above = self.pair_above and 1 or -1

        two_objects_left = self.two_objects_left
        if two_objects_left is None:
            two_objects_left = self.rng.uniform() > 0.5

        left_target_offset = np.array((pair_above * half_distance, left_target_horizontal_offset), dtype=np.int)
        right_target_offset = np.array((pair_above * half_distance, right_target_horizontal_offset), dtype=np.int)
        other_side_target_offset = np.array((-pair_above * half_distance, two_objects_left and left_target_horizontal_offset or right_target_horizontal_offset), dtype=np.int)

        if two_objects_left:
            target_offsets = (left_target_offset, right_target_offset, other_side_target_offset)
        else: 
            target_offsets = (right_target_offset, left_target_offset, other_side_target_offset)
            
        target_positions = [tuple(reference_center_position + offset) for offset in target_offsets]

        if self.two_reference_objects:
            second_reference_center = np.copy(reference_center_position)
            two_targets_between = self.two_targets_between and 1 or -1
            second_reference_vertical_direction = pair_above * two_targets_between
            second_reference_center[0] += second_reference_vertical_direction * distance
            reference_center_position = [reference_center_position, second_reference_center]
            
        if self.n_target_types == 1:
            target_indices = (self.rng.integers(0, self.stimulus_generator.n_target_types), ) * 3
        if self.n_target_types == 2:
            pair_color, single_color = self.rng.choice(np.arange(self.stimulus_generator.n_target_types),
                                                       size=2, replace=False)
#             pair_color = self.rng.uniform() > 0.5
#             single_color = 1 - pair_color
            target_indices = (single_color, pair_color, pair_color)
            
        elif self.n_target_types == 3:
            target_indices = [0, 1, 2]
            self.rng.shuffle(target_indices)
            target_indices = tuple(target_indices)

        return self.stimulus_generator.batch_generate(target_positions, 
                                                      reference_center_position, 
                                                      target_indices, 
                                                      normalize=normalize,
                                                      transpose_target=self.transpose)
