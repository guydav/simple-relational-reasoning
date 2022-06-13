import numpy as np
import torch
import tqdm

from abc import abstractmethod
import typing
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
DEFAULT_MARGIN_BUFFER = 2

class AbstractTripletGenerator:
    def __init__(self, stimulus_generator, transpose=False, seed=DEFAULT_RANDOM_SEED, use_tqdm=False):
        self.stimulus_generator = stimulus_generator
        self.transpose = transpose
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


class TripletGenerator(AbstractTripletGenerator):
    def __init__(self, stimulus_generator, relation,
                 two_reference_objects=False, two_targets_between=True, n_target_types=1,
                 transpose=False, vertical_margin=0, horizontal_margin=0, seed=DEFAULT_RANDOM_SEED, use_tqdm=False):
        super().__init__(stimulus_generator, transpose=transpose, seed=seed, use_tqdm=use_tqdm)
        
        if relation == BETWEEN_RELATION and not two_reference_objects:
            raise ValueError('Between relation requires two reference objects')
        
        self.stimulus_generator = stimulus_generator
        self.relation = relation
        self.two_reference_objects = two_reference_objects
        self.two_targets_between = two_targets_between
        self.n_target_types = n_target_types
        
        if n_target_types > self.stimulus_generator.n_target_types:
            raise ValueError(f'Expected n_target_types={n_target_types} <= self.stimulus_generator.n_target_types={self.stimulus_generator.n_target_types}')
        
        self.vertical_margin = vertical_margin
        self.horizontal_margin = horizontal_margin
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.use_tqdm = use_tqdm
    
     
class QuinnTripletGenerator(TripletGenerator):
    def __init__(self, stimulus_generator, distance_endpoints, relation,
                 pair_above=None, two_objects_left=None,
                 two_reference_objects=False, 
                 two_targets_between=True, 
                 adjacent_reference_objects=False,
                 n_target_types=1, transpose=False,
                 vertical_margin=0, horizontal_margin=0,
                 margin_buffer=DEFAULT_MARGIN_BUFFER,
                 n_habituation_stimuli=1,
                 multiple_habituation_radius=DEFAULT_MULTIPLE_HABITUATION_RADIUS,
                 seed=DEFAULT_RANDOM_SEED, use_tqdm=False, track_centroids=False):
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
        self.margin_buffer = margin_buffer
        self.n_habituation_stimuli = n_habituation_stimuli
        self.multiple_habituation_radius = multiple_habituation_radius

        self.reference_width = self.stimulus_generator.reference_size[1]
        self.reference_height = self.stimulus_generator.reference_size[0]
        self.target_width = self.stimulus_generator.target_size[1]
        self.target_height = self.stimulus_generator.target_size[0]

        self.track_centroids = track_centroids
        if self.track_centroids:
            self.stimulus_centroids = []
    
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
                max_inter_reference_distance = target_distance - (self.reference_height + self.target_height + self.margin_buffer * 4)
                if half_target_distance >= max_inter_reference_distance:
                    inter_reference_distance = max_inter_reference_distance
                else:
                    inter_reference_distance = self.rng.integers(half_target_distance, max_inter_reference_distance)
            # else:  # Same distance in above/below and between
            #     inter_reference_distance = target_distance

        stimulus_centroid_position = np.array([self.stimulus_generator.canvas_size[0] // 2,
             self.stimulus_generator.canvas_size[1] // 2], dtype=np.int)

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

        stimulus, centroid =  self.stimulus_generator.batch_generate(target_positions, 
                                                      reference_positions, 
                                                      target_indices, 
                                                      normalize=normalize,
                                                      transpose_target=self.transpose,
                                                      return_centroid=True)

        if self.track_centroids:
            self.stimulus_centroids.append(centroid)

        return stimulus


TOP_LEFT = 'top_left'
TOP_RIGHT = 'top_right'
BOTTOM_LEFT = 'bottom_left'
BOTTOM_RIGHT = 'bottom_right'
FLAGS_TO_QUADRANT_MAP = {
    (True, True): TOP_LEFT,
    (True, False): TOP_RIGHT,
    (False, True): BOTTOM_LEFT,
    (False, False): BOTTOM_RIGHT,
}

class NoReferenceEquidistantTripletGenerator(AbstractTripletGenerator):
    def __init__(self, stimulus_generator, 
                 n_target_types=1, 
                 margin_buffer=DEFAULT_MARGIN_BUFFER,
                 n_habituation_stimuli=1,
                 multiple_habituation_radius=DEFAULT_MULTIPLE_HABITUATION_RADIUS,
                 seed=DEFAULT_RANDOM_SEED, use_tqdm=False, track_habituation_positions=False):
        super(NoReferenceEquidistantTripletGenerator, self).__init__(
            stimulus_generator=stimulus_generator, transpose=False, seed=seed, use_tqdm=use_tqdm)
            
        self.n_target_types = n_target_types
        self.margin_buffer = margin_buffer
        self.n_habituation_stimuli = n_habituation_stimuli
        self.multiple_habituation_radius = multiple_habituation_radius

        self.track_habituation_positions = track_habituation_positions
        if self.track_habituation_positions:
            self.habituation_positions = []

        self.target_width = self.stimulus_generator.target_size[1]
        self.target_height = self.stimulus_generator.target_size[0]

        self.default_margin = (self.target_height // 2 + self.margin_buffer, self.target_width // 2 + self.margin_buffer)
        if self.n_habituation_stimuli > 1:
            self.default_margin = (self.default_margin[0] + multiple_habituation_radius, self.default_margin[1] + multiple_habituation_radius)
        self.canvas_size = self.stimulus_generator.canvas_size
        self.middle_row_index = self.canvas_size[0] // 2
        self.middle_col_index = self.canvas_size[1] // 2

        self.quadrant_to_endpoints = {
            TOP_LEFT: ((0, self.middle_row_index), (0, self.middle_col_index)),
            TOP_RIGHT: ((0, self.middle_row_index), (self.middle_col_index, self.canvas_size[1])),
            BOTTOM_LEFT: ((self.middle_row_index, self.canvas_size[0]), (0, self.middle_col_index)),
            BOTTOM_RIGHT: ((self.middle_row_index, self.canvas_size[0]), (self.middle_col_index, self.canvas_size[1])),
        }
    
    def _sample_point_in_quadrant(self, other_side_object_up: float, pair_left: float, 
        margin: typing.Optional[typing.Tuple[int, int]] = None):
        if margin is None:
            margin = self.default_margin

        quadrant = self._get_quadrant(other_side_object_up, pair_left)
        row_margin, col_margin = margin
        row_endpoints, col_endpoints = self.quadrant_to_endpoints[quadrant]
        return (self._sample_point_with_margin(row_endpoints, row_margin),
                self._sample_point_with_margin(col_endpoints, col_margin))

    def _get_quadrant(self, other_side_object_up, pair_left):
        return FLAGS_TO_QUADRANT_MAP[(other_side_object_up > 0, pair_left > 0)]

    def _sample_point_with_margin(self, endpoints: typing.Tuple[int, int], margin: int = 0):
        return self.rng.integers(endpoints[0] + margin, endpoints[1] - margin)

    def generate_single_triplet(self, normalize=True):
        other_side_object_up = np.sign(self.rng.uniform(-0.5, 0.5))
        pair_left = np.sign(self.rng.uniform(-0.5, 0.5))

        habituation_position, target_distance = self._sample_target_positions(other_side_object_up, pair_left)
        target_positions = [np.copy(habituation_position) for _ in range(3)]
        target_positions[2][1] += target_distance * pair_left
        target_positions[1][0] += target_distance * other_side_object_up

        if self.n_habituation_stimuli > 1:
            angle_step = 360 // self.n_habituation_stimuli
            habituation_angles = np.arange(0, 360, angle_step)
            angle_offset = self.rng.integers(0, angle_step)
            habituation_angles = (habituation_angles + angle_offset) % 360

            habituation_positions = []
            for angle in habituation_angles:
                angle_vec = np.array([-np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))])
                habituation_positions.append((habituation_position + (angle_vec * self.multiple_habituation_radius)).astype(np.int))

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

        if self.track_habituation_positions:
            self.habituation_positions.append(habituation_position)
        return self.stimulus_generator.batch_generate(target_positions, 
                                                      [], 
                                                      target_indices, 
                                                      normalize=normalize,
                                                      transpose_target=False,
                                                      return_centroid=False,
                                                      pad_and_crop=False)

    def _sample_target_positions(self, other_side_object_up, pair_left):
        habituation_position, target_distance = None, None

        while target_distance is None:
            habituation_position = self._sample_point_in_quadrant(other_side_object_up, pair_left)
        
            row_min_distance = abs(self.middle_row_index - habituation_position[0])
            row_max_distance = row_min_distance + self.middle_col_index

            col_min_distance = abs(self.middle_col_index - habituation_position[1])
            col_max_distance = col_min_distance + self.middle_col_index

            min_distance = max(row_min_distance, col_min_distance)
            max_distance = min(row_max_distance, col_max_distance)
            max_margin = max(self.default_margin)

            if max_distance <= min_distance + (max_margin * 2):
                continue

            target_distance = self._sample_point_with_margin((min_distance, max_distance), margin=max_margin)

        return habituation_position, target_distance



class NoReferenceDiagonalTripletGenerator(NoReferenceEquidistantTripletGenerator):
    def __init__(self, stimulus_generator, 
                    n_target_types=1, 
                    margin_buffer=DEFAULT_MARGIN_BUFFER,
                    n_habituation_stimuli=1,
                    multiple_habituation_radius=DEFAULT_MULTIPLE_HABITUATION_RADIUS,
                    seed=DEFAULT_RANDOM_SEED, use_tqdm=False, track_habituation_positions=False):

        super().__init__(stimulus_generator, 
            n_target_types=n_target_types, margin_buffer=margin_buffer,
            n_habituation_stimuli=n_habituation_stimuli, 
            multiple_habituation_radius=multiple_habituation_radius,
            seed=seed, use_tqdm=use_tqdm, track_habituation_positions=track_habituation_positions)

    def _sample_target_positions(self, other_side_object_up, pair_left):
        max_margin = max(self.default_margin)
        diagonal_coord = self.rng.integers(max_margin, self.middle_row_index - max_margin)

        habituation_position = (int((other_side_object_up * diagonal_coord) % self.canvas_size[0]), int((pair_left * diagonal_coord) % self.canvas_size[1]))
        target_distance = self.canvas_size[0] - 2 * diagonal_coord
        return habituation_position, target_distance
    

class SameHalfTripletGenerator(NoReferenceEquidistantTripletGenerator):
    def __init__(self, stimulus_generator, same_horizontal_half=True,
                    n_target_types=1, 
                    margin_buffer=DEFAULT_MARGIN_BUFFER,
                    n_habituation_stimuli=1,
                    multiple_habituation_radius=DEFAULT_MULTIPLE_HABITUATION_RADIUS,
                    seed=DEFAULT_RANDOM_SEED, use_tqdm=False, track_habituation_positions=False):

        super().__init__(stimulus_generator, 
            n_target_types=n_target_types, margin_buffer=margin_buffer,
            n_habituation_stimuli=n_habituation_stimuli, 
            multiple_habituation_radius=multiple_habituation_radius,
            seed=seed, use_tqdm=use_tqdm, track_habituation_positions=track_habituation_positions)

        self.same_horizontal_half = same_horizontal_half

    def _sample_target_positions(self, other_side_object_up, pair_left):
        habituation_position, target_distance = None, None

        while target_distance is None:
            habituation_position = self._sample_point_in_quadrant(other_side_object_up, pair_left)

            max_margin = max(self.default_margin)
        
            if self.same_horizontal_half:
                row_min_distance = abs(self.middle_row_index - habituation_position[0])
                row_max_distance = row_min_distance + self.middle_col_index

                col_min_distance = 2 * max_margin
                col_max_distance = abs(self.middle_col_index - habituation_position[1])

            else:  # same vertical half
                row_min_distance = 2 * max_margin
                row_max_distance = abs(self.middle_row_index - habituation_position[0])

                col_min_distance = abs(self.middle_col_index - habituation_position[1])
                col_max_distance = col_min_distance + self.middle_col_index

            min_distance = max(row_min_distance, col_min_distance)
            max_distance = min(row_max_distance, col_max_distance)
            max_margin = max(self.default_margin)

            if max_distance <= min_distance + (max_margin * 2):
                continue

            target_distance = self._sample_point_with_margin((min_distance, max_distance), margin=max_margin)

        return habituation_position, target_distance


class SameQuadrantTripletGenerator(NoReferenceEquidistantTripletGenerator):
    def __init__(self, stimulus_generator, 
                    n_target_types=1, 
                    margin_buffer=DEFAULT_MARGIN_BUFFER,
                    n_habituation_stimuli=1,
                    multiple_habituation_radius=DEFAULT_MULTIPLE_HABITUATION_RADIUS,
                    seed=DEFAULT_RANDOM_SEED, use_tqdm=False, track_habituation_positions=False):

        super().__init__(stimulus_generator, 
            n_target_types=n_target_types, margin_buffer=margin_buffer,
            n_habituation_stimuli=n_habituation_stimuli, 
            multiple_habituation_radius=multiple_habituation_radius,
            seed=seed, use_tqdm=use_tqdm, track_habituation_positions=track_habituation_positions)

    def _sample_target_positions(self, other_side_object_up, pair_left):
        habituation_position, target_distance = None, None

        while target_distance is None:
            habituation_position = self._sample_point_in_quadrant(other_side_object_up, pair_left)
            max_margin = max(self.default_margin)

            row_min_distance = 2 * max_margin
            row_max_distance = abs(self.middle_row_index - habituation_position[0])

            col_min_distance = 2 * max_margin
            col_max_distance = abs(self.middle_col_index - habituation_position[1])

            min_distance = max(row_min_distance, col_min_distance)
            max_distance = min(row_max_distance, col_max_distance)
            max_margin = max(self.default_margin)

            if max_distance <= min_distance + (max_margin * 2):
                continue

            target_distance = self._sample_point_with_margin((min_distance, max_distance), margin=max_margin)

        return habituation_position, target_distance


DEFAULT_TARGET_STEP = 2

class TSNEStimuliSetGenerator(TripletGenerator):
    def __init__(self, stimulus_generator, distance_endpoints, relation,
                 target_step=DEFAULT_TARGET_STEP,
                 two_reference_objects=False, 
                 two_targets_between=True, 
                 adjacent_reference_objects=False,
                 fixed_inter_reference_distance=None,
                 fixed_target_index=None,
                 center_stimuli=True,
                 transpose=False,
                 vertical_margin=0, horizontal_margin=0,
                 margin_buffer=DEFAULT_MARGIN_BUFFER,
                 seed=DEFAULT_RANDOM_SEED, use_tqdm=False, track_centroids=False):
        super(TSNEStimuliSetGenerator, self).__init__(
            stimulus_generator=stimulus_generator, relation=relation,
            two_reference_objects=two_reference_objects,
            two_targets_between=two_targets_between, 
            n_target_types=1,
            transpose=transpose, vertical_margin=vertical_margin, 
            horizontal_margin=horizontal_margin, seed=seed, use_tqdm=use_tqdm)
        
        if not hasattr(distance_endpoints, '__len__'):
            distance_endpoints = (distance_endpoints, distance_endpoints)        
            
        self.distance_endpoints = distance_endpoints
        self.target_step = target_step
        self.adjacent_reference_objects = adjacent_reference_objects
        self.fixed_inter_reference_distance = fixed_inter_reference_distance
        self.fixed_target_index = fixed_target_index
        self.center_stimuli = center_stimuli
        self.margin_buffer = margin_buffer

        self.reference_width = self.stimulus_generator.reference_size[1]
        self.reference_height = self.stimulus_generator.reference_size[0]
        self.target_width = self.stimulus_generator.target_size[1]
        self.target_height = self.stimulus_generator.target_size[0]

        self.track_centroids = track_centroids
        if self.track_centroids:
            self.stimulus_centroids = []
    
    def _left_right_limits(self, reference_position):
        left = reference_position[1] - (self.reference_width // 2) + (self.target_width // 2) + self.margin_buffer
        right = reference_position[1] + (self.reference_width // 2) - (self.target_width // 2) - self.margin_buffer
        return left,right

    def _limits_to_positions(self, top, bottom, left, right):
        # returns a X by 2 array of positions
                return np.mgrid[top:bottom:self.target_step, left:right:self.target_step].reshape(2, -1).T

    def _tile_above(self, reference_position):
        top = max(reference_position[0] - self.distance_endpoints[1], self.target_height // 2 + self.margin_buffer)
        bottom = reference_position[0] - (self.reference_height // 2) - (self.target_height // 2) - self.margin_buffer
        left, right = self._left_right_limits(reference_position)

        return self._limits_to_positions(top, bottom, left, right)

    def _tile_below(self, reference_position):
        top = reference_position[0] + (self.reference_height // 2) + (self.target_height // 2) + self.margin_buffer
        bottom = min(reference_position[0] + self.distance_endpoints[1], 
            self.stimulus_generator.canvas_size[0] - (self.target_height // 2) - self.margin_buffer)
        left, right = self._left_right_limits(reference_position)

        return self._limits_to_positions(top, bottom, left, right)

    def _tile_between(self, top_reference_position, bottom_reference_position):
        top = top_reference_position[0] + (self.reference_height // 2) + (self.target_height // 2) + self.margin_buffer
        bottom = bottom_reference_position[0] - (self.reference_height // 2) - (self.target_height // 2) - self.margin_buffer
        left, right = self._left_right_limits(top_reference_position)

        return self._limits_to_positions(top, bottom, left, right)

    def generate_single_triplet(self, normalize=True):
        distance_endpoints = self.distance_endpoints

        target_distance = self.rng.integers(*distance_endpoints)
        half_target_distance = target_distance // 2

        inter_reference_distance = 0
        if self.two_reference_objects:
            if self.adjacent_reference_objects:
                inter_reference_distance = self.reference_height * 1.5
            
            elif self.relation == BETWEEN_RELATION:
                if self.fixed_inter_reference_distance is not None and distance_endpoints[0] <= self.fixed_inter_reference_distance <= distance_endpoints[1]:
                    inter_reference_distance = self.fixed_inter_reference_distance
                else:
                    inter_reference_distance = target_distance
            
            else: # self.relation == ABOVE_BELOW_RELATION
                max_inter_reference_distance = target_distance - (self.reference_height + self.target_height + self.margin_buffer * 4)
                if self.fixed_inter_reference_distance is not None and self.fixed_inter_reference_distance < max_inter_reference_distance:
                    inter_reference_distance = self.fixed_inter_reference_distance
                if half_target_distance >= max_inter_reference_distance:
                    inter_reference_distance = max_inter_reference_distance
                else:
                    inter_reference_distance = self.rng.integers(half_target_distance, max_inter_reference_distance)
            # else:  # Same distance in above/below and between
            #     inter_reference_distance = target_distance

        stimulus_centroid_position = np.array([self.stimulus_generator.canvas_size[0] // 2,
             self.stimulus_generator.canvas_size[1] // 2], dtype=np.int)

        # compute reference positions relative to the centroid
        reference_positions = []
        all_target_positions = []

        if self.two_reference_objects:
            reference_positions.append(np.copy(stimulus_centroid_position))
            reference_positions.append(np.copy(stimulus_centroid_position))
            reference_positions[0][0] -= inter_reference_distance // 2
            reference_positions[1][0] += inter_reference_distance // 2

            all_target_positions.append(self._tile_above(reference_positions[0]))
            if self.relation == BETWEEN_RELATION:
                all_target_positions.append(self._tile_between(reference_positions[0], reference_positions[1]))
            all_target_positions.append(self._tile_below(reference_positions[1]))
            

        else:  # self.relation == ABOVE_BELOW_RELATION and not self.two_reference_objects:
            reference_positions.append(np.copy(stimulus_centroid_position))
            all_target_positions.append(self._tile_above(reference_positions[0]))
            all_target_positions.append(self._tile_below(reference_positions[0]))

        target_positions = list(np.concatenate(all_target_positions))    
        if self.fixed_target_index is not None and 0 <= self.fixed_target_index < self.stimulus_generator.n_target_types:
            target_index = self.fixed_target_index

        else:
            target_index = self.rng.integers(0, self.stimulus_generator.n_target_types)

        target_indices = [target_index] * len(target_positions)

        stimulus, centroid =  self.stimulus_generator.batch_generate(target_positions, 
                                                      reference_positions, 
                                                      target_indices, 
                                                      normalize=normalize,
                                                      transpose_target=self.transpose,
                                                      return_centroid=True,
                                                      crop_to_center=self.center_stimuli)

        if self.track_centroids:
            self.stimulus_centroids.append(centroid)

        return stimulus



TRIPLET_GENERATORS = {
    'quinn': QuinnTripletGenerator,
    'equidistant': NoReferenceEquidistantTripletGenerator,
    'diagonal': NoReferenceDiagonalTripletGenerator,
    'same_half': SameHalfTripletGenerator,
    'same_quadrant': SameQuadrantTripletGenerator,
    'tsne': TSNEStimuliSetGenerator
}