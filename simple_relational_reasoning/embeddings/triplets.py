import numpy as np
import torch
import tqdm

from abc import abstractmethod
from functools import lru_cache
from tqdm.notebook import tqdm


DEFAULT_RANDOM_SEED = 33
TRIPLET_CACHE_SIZE = 16

ABOVE_BELOW_RELATION = 'above_below'
BETWEEN_RELATION = 'between'

RELATIONS = (
    ABOVE_BELOW_RELATION,
    BETWEEN_RELATION,
)

DEFAULT_MULTIPLE_HABITUATION_RADIUS = 10
class TripletGenerator:
    def __init__(self, stimulus_generator, relation,
                 two_reference_objects=False, two_targets_between=True, n_target_types=1,
                 transpose=False, vertical_margin=0, horizontal_margin=0, seed=DEFAULT_RANDOM_SEED, use_tqdm=False):
        
        if relation == BETWEEN_RELATION and not two_reference_objects:
            raise ValueError('Between relation requires two reference objects')
        
        self.stimulus_generator = stimulus_generator
        self.relation = relation
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
     
class QuinnTripletGenerator(TripletGenerator):
    def __init__(self, stimulus_generator, distance_endpoints, relation,
                 pair_above=None, two_objects_left=None,
                 two_reference_objects=False, 
                 two_targets_between=True, 
                 adjacent_reference_objects=False,
                 n_target_types=1, transpose=False,
                 vertical_margin=0, horizontal_margin=0,
                 extra_diagonal_margin=0,
                 n_habituation_stimuli=1,
                 multiple_habituation_radius=DEFAULT_MULTIPLE_HABITUATION_RADIUS,
                 seed=DEFAULT_RANDOM_SEED, use_tqdm=False):
        super(QuinnTripletGenerator, self).__init__(
            stimulus_generator=stimulus_generator, relation=relation,
            two_reference_objects=two_reference_objects,
            two_targets_between=two_targets_between, 
            n_target_types=n_target_types,
            transpose=transpose, vertical_margin=vertical_margin, 
            horizontal_margin=horizontal_margin, seed=seed, use_tqdm=use_tqdm)
        
        if not hasattr(distance_endpoints, '__len__'):
            distance_endpoints = (distance_endpoints, distance_endpoints)        
            
        self.distance_endpoints = distance_endpoints
        self.adjacent_reference_objects = adjacent_reference_objects
        self.pair_above = pair_above
        self.two_objects_left = two_objects_left
        self.extra_diagonal_margin = extra_diagonal_margin
        self.n_habituation_stimuli = n_habituation_stimuli
        self.multiple_habituation_radius = multiple_habituation_radius

        self.centroids = []

        self.reference_width = self.stimulus_generator.reference_size[1]
        self.reference_height = self.stimulus_generator.reference_size[0]
        self.target_width = self.stimulus_generator.target_size[1]
        self.target_height = self.stimulus_generator.target_size[0]
    
    def generate_single_triplet(self, normalize=True):
        distance_endpoints = self.distance_endpoints
        if self.n_habituation_stimuli > 1:
            distance_endpoints = (distance_endpoints[0] + self.multiple_habituation_radius, distance_endpoints[1])

        target_distance = self.rng.integers(*distance_endpoints)
        half_target_distance = target_distance // 2

        inter_reference_distance = 0
        if self.two_reference_objects:
            if self.adjacent_reference_objects:
                inter_reference_distance = self.reference_height * 1.5
            elif self.relation == BETWEEN_RELATION:
                inter_reference_distance = target_distance
            else: # self.relation == ABOVE_BELOW_RELATION
                inter_reference_distance = half_target_distance

        # compute the margins
        min_horizontal_margin = self.stimulus_generator.reference_size[1] // 2 + 3

        if self.relation == BETWEEN_RELATION:
            min_vertical_margin = target_distance + (self.reference_height // 2) + 3
            
        else:  # self.relation == ABOVE_BELOW_RELATION:
            min_vertical_margin = half_target_distance + (self.reference_height // 2) + 3

        # account for wonkiness in rotation stimuli and ending out of frame
        if self.extra_diagonal_margin and self.stimulus_generator.rotate_angle is not None:
            angle_sin = np.sin(np.deg2rad(self.stimulus_generator.rotate_angle))
            min_vertical_margin += self.extra_diagonal_margin * angle_sin
            # min_horizontal_margin = self.extra_diagonal_margin * angle_sin / 2

        # account for the multuiple habituation stimuli radius 
        if self.n_habituation_stimuli > 1:
            min_vertical_margin += self.multiple_habituation_radius
            min_horizontal_margin += self.multiple_habituation_radius

        vertical_margin = max(min_vertical_margin, self.vertical_margin)
        horizontal_margin = max(min_horizontal_margin, self.horizontal_margin)

        stimulus_centroid_position = np.array(
            (self.rng.integers(vertical_margin, self.stimulus_generator.canvas_size[0] - vertical_margin),
             self.rng.integers(horizontal_margin, self.stimulus_generator.canvas_size[1] - horizontal_margin)), 
            dtype=np.int)

        # compute reference positions relative to the centroid
        reference_positions = []

        if self.two_reference_objects:
            reference_positions.append(np.copy(stimulus_centroid_position))
            reference_positions.append(np.copy(stimulus_centroid_position))
            reference_positions[0][0] -= inter_reference_distance // 2
            reference_positions[1][0] += inter_reference_distance // 2

        else:  # self.relation == ABOVE_BELOW_RELATION and not self.two_reference_objects:
            reference_positions.append(np.copy(stimulus_centroid_position))

        # compute target positions relative to centroid -- trivial for above/below and between, hard for diagonal
        if self.pair_above is None:
            pair_above = np.sign(self.rng.uniform(-0.5, 0.5))
        else:
            pair_above = self.pair_above and 1 or -1

        target_positions = []
        
        target_horizontal_margin = (self.reference_width - self.target_width) // 2
        left_target_horizontal_offset = self.rng.integers(-target_horizontal_margin, target_horizontal_margin - target_distance)
        right_target_horizontal_offset = left_target_horizontal_offset + target_distance
        
        two_objects_left = self.two_objects_left
        if two_objects_left is None:
            two_objects_left = self.rng.uniform() > 0.5

        target_positions.append(np.copy(stimulus_centroid_position))
        target_positions.append(np.copy(stimulus_centroid_position))

        target_positions[0][1] += left_target_horizontal_offset
        target_positions[1][1] += right_target_horizontal_offset

        if not two_objects_left:
            target_positions = target_positions[::-1]

        # for between, no need to shift the first two targets vertically
        # third target is either above or below, and either above the left or above the right
        # shifted up/down by the target distance (if between/outside) or half the distance (if above/below)

        third_target_position = np.copy(target_positions[0])
        third_target_position[0] += -pair_above * (target_distance if self.relation == BETWEEN_RELATION else half_target_distance)
        target_positions.append(third_target_position)

        if self.relation == ABOVE_BELOW_RELATION:
            # shift the first two targets vertically by half the target distance
            target_positions[0][0] += pair_above * half_target_distance
            target_positions[1][0] += pair_above * half_target_distance

        # at this point, target_positions[0] is the habituation target, so if I need to create multiple ones, do it here:
        if self.n_habituation_stimuli > 1:
            habituation_centroid = target_positions[0]
            angle_step = 360 // self.n_habituation_stimuli
            habituation_angles = np.arange(0, 360, angle_step)
            angle_offset = self.rng.integers(0, angle_step)
            habituation_angles = (habituation_angles + angle_offset) % 360

            habituation_positions = []
            for angle in habituation_angles:
                angle_vec = np.array([-np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))])
                habituation_positions.append((habituation_centroid + (angle_vec * self.multiple_habituation_radius)).astype(np.int))

            habituation_positions.extend(target_positions[1:])
            target_positions = habituation_positions

        if self.n_target_types == 1:
            target_indices = [self.rng.integers(0, self.stimulus_generator.n_target_types)] * len(target_positions)

        if self.n_target_types == 2:
            # can we allocate a unique target to each stimulus?
            if self.stimulus_generator.n_target_types > self.n_habituation_stimuli > 1:
                target_indices = self.rng.choice(np.arange(self.stimulus_generator.n_target_types),
                    size=self.n_habituation_stimuli + 1, replace=False)

                target_indices = list(target_indices[:-1]) + [target_indices[-1]] * 2

            else:  # either only one habituation stimulus or not enough target types for unique habituation targets
                pair_color, single_color = self.rng.choice(np.arange(self.stimulus_generator.n_target_types),
                    size=2, replace=False)
                target_indices = [single_color] * self.n_habituation_stimuli + [pair_color] * 2
            
        elif self.n_target_types == 3:
            target_indices = self.rng.choice(np.arange(self.stimulus_generator.n_target_types),
                size=3, replace=False)
            target_indices = [target_indices[0]] * self.n_habituation_stimuli + target_indices[1:]

        self.centroids.append(stimulus_centroid_position)

        return self.stimulus_generator.batch_generate(target_positions, 
                                                      reference_positions, 
                                                      target_indices, 
                                                      normalize=normalize,
                                                      transpose_target=self.transpose,
                                                      stimulus_centroid=stimulus_centroid_position)

TRIPLET_GENERATORS = {
    'quinn': QuinnTripletGenerator,
}