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
DIAGONAL_RELATION = 'diagonal'

RELATIONS = (
    ABOVE_BELOW_RELATION,
    BETWEEN_RELATION,
    DIAGONAL_RELATION
)
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
    

class EquilateralTripletGenerator(TripletGenerator):
    def __init__(self, stimulus_generator, side_length_endpoints, relation,
                 pair_above=None,
                 two_reference_objects=False, two_targets_between=True, 
                 n_target_types=1, transpose=False, 
                 vertical_margin=0, horizontal_margin=0, 
                 seed=DEFAULT_RANDOM_SEED, use_tqdm=False):
        super(EquilateralTripletGenerator, self).__init__(
            stimulus_generator=stimulus_generator, relation=relation,
            two_reference_objects=two_reference_objects,
            two_targets_between=two_targets_between, n_target_types=n_target_types,
            transpose=transpose, vertical_margin=vertical_margin,
            horizontal_margin=horizontal_margin, seed=seed, use_tqdm=use_tqdm)
        
        if not hasattr(side_length_endpoints, '__len__'):
            side_length_endpoints = (side_length_endpoints, side_length_endpoints)
            
        self.side_length_endpoints = side_length_endpoints
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
    def __init__(self, stimulus_generator, distance_endpoints, relation,
                 pair_above=None, two_objects_left=None,
                 two_reference_objects=False, 
                 two_targets_between=True, 
                 adjacent_reference_objects=False,
                 n_target_types=1, transpose=False,
                 vertical_margin=0, horizontal_margin=0,
                 diagonal_margin_buffer=10,
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
        self.diagonal_margin_buffer = diagonal_margin_buffer

        self.reference_width = self.stimulus_generator._reference_object().shape[2]
        self.reference_height = self.stimulus_generator._reference_object().shape[1]
        self.target_width = self.stimulus_generator.target_size[1]
        self.target_height = self.stimulus_generator.target_size[0]
    
    def generate_single_triplet(self, normalize=True):
        target_distance = self.rng.integers(*self.distance_endpoints)
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
        min_horizontal_margin = self.reference_width // 2 + 1

        if self.relation == DIAGONAL_RELATION:
            min_horizontal_margin += self.diagonal_margin_buffer
            min_vertical_margin = self.reference_height // 2 + 1 + self.diagonal_margin_buffer

        elif self.relation == BETWEEN_RELATION:
            min_vertical_margin = target_distance + (self.reference_height // 2) + 1
            
        else:  # self.relation == ABOVE_BELOW_RELATION:
            min_vertical_margin = half_target_distance + (self.reference_height // 2) + 1

        vertical_margin = max(min_vertical_margin, self.vertical_margin)
        horizontal_margin = max(min_horizontal_margin, self.horizontal_margin)

        stimulus_centroid_position = np.array(
            (self.rng.integers(vertical_margin, self.stimulus_generator.canvas_size[0] - vertical_margin),
             self.rng.integers(horizontal_margin, self.stimulus_generator.canvas_size[1] - horizontal_margin)), 
            dtype=np.int)

        # compute reference positions relative to the centroid
        reference_positions = []
        if self.relation == DIAGONAL_RELATION:
            reference_positions.append(np.copy(stimulus_centroid_position))

        elif self.two_reference_objects:
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
            
        # TODO: consider the case of multiple habituation stimuli
        # this is not actually that hard -- 
        # (1) make sure that the patch stimulus generator accepts an arbitrary number of targets
        # (2) fix the target indices generated at the end of this function to be identical 
        # all of the habituation stimuli
        # (3) accept a parameter for the radius on which we're placing the targets around the sampled position
        # (4) increase the margin from the edge by this radius
        # (5) sample positions uniformly on the circle around that radius
        # making sure to also take into account the `two_objects_left` parameter
        # (6) in the above case for diagonal, first check which one we can place across, then place the multiple stimuli

        # TODO: after doing the above, go to the task implementation and make sure that won't break

        target_positions = []

        if self.relation == DIAGONAL_RELATION:
            refernce_angle = self.stimulus_generator.rotate_angle

            if refernce_angle is None:
                raise ValueError('Diagonal relation requires the stimulus generation have a rotation angle')            

            if np.abs(refernce_angle) >= 90:
                raise ValueError('Diagonal relation requires the stimulus generation have a rotation angle between -90 and 90')

            approx_radius = np.mean((self.target_height, self.target_width)) / 2
            sin_angle = np.sin(np.deg2rad(refernce_angle))
            cos_angle = np.cos(np.deg2rad(refernce_angle))
            angle_vec = np.array([-sin_angle, cos_angle])  # (row, column) rather than (x, y), and rows increase down, rather than up
            cot_angle = cos_angle / sin_angle
            
            # slightly annoying geometry to figure out how far left/right on the referrence we can place target above/below
            if pair_above == (refernce_angle > 0):
                left_target_min_margin = - cot_angle * (half_target_distance - approx_radius)
                right_target_min_margin = cot_angle * (half_target_distance + approx_radius)

            else:
                left_target_min_margin = cot_angle * (half_target_distance - approx_radius)
                right_target_min_margin = - cot_angle * (half_target_distance + approx_radius)
                
            left_target_horizontal_offset = self.rng.integers(
                int(-self.reference_width / 2 + left_target_min_margin), 
                int(self.reference_width / 2 + right_target_min_margin - target_distance))

            right_target_horizontal_offset = left_target_horizontal_offset + target_distance

            # TODO: compute the actual positions for the two points whose offsets we just computed
            left_position_on_reference = stimulus_centroid_position + left_target_horizontal_offset * angle_vec
            right_position_on_reference = stimulus_centroid_position + right_target_horizontal_offset * angle_vec

            left_target_position, right_target_position = self._project_positions(half_target_distance, pair_above, angle_vec, left_position_on_reference, right_position_on_reference)

            left_potential_position, right_potential_position = self._project_positions(half_target_distance, -1 * pair_above, angle_vec, left_position_on_reference, right_position_on_reference)

            reference_box_corner_1 = stimulus_centroid_position - self.reference_width / 2 * angle_vec
            reference_box_corner_2 = stimulus_centroid_position + self.reference_width / 2 * angle_vec
            
            row_limits = (min(reference_box_corner_1[0], reference_box_corner_2[0]), max(reference_box_corner_1[0], reference_box_corner_2[0]))
            col_limits = (min(reference_box_corner_1[1], reference_box_corner_2[1]), max(reference_box_corner_1[1], reference_box_corner_2[1]))

            left_potential_valid = self._check_point_in_limits(left_potential_position, row_limits, col_limits)
            right_potential_valid = self._check_point_in_limits(right_potential_position, row_limits, col_limits)

            use_left = None
            if left_potential_valid and right_potential_valid:
                use_left = self.rng.uniform() < 0.5
            elif left_potential_valid:
                use_left = True
            elif right_potential_valid:
                use_left = False
            else:
                use_left = np.linalg.norm(left_potential_position - stimulus_centroid_position) < np.linalg.norm(right_potential_position - stimulus_centroid_position)
                
            if use_left:
                target_positions = [left_target_position, right_target_position, left_potential_position]
            else:
                target_positions = [right_target_position, left_target_position, right_potential_position]

            

        else:
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

            # for between, no need to shift the first two targets vertically
            # third target is either above or below, and either above the left or above the right
            # shifted up/down by the target distance (if between/outside) or half the distance (if above/below)
            third_target_position = np.copy(target_positions[0] if two_objects_left else target_positions[1])
            third_target_position[0] += -pair_above * (target_distance if self.relation == BETWEEN_RELATION else half_target_distance)
            target_positions.append(third_target_position)

            if self.relation == ABOVE_BELOW_RELATION:
                # shift the first two targets vertically by half the target distance
                target_positions[0][0] += pair_above * half_target_distance
                target_positions[1][0] += pair_above * half_target_distance

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
            # self.rng.shuffle(target_indices)
            target_indices = tuple(target_indices)

        return self.stimulus_generator.batch_generate(target_positions, 
                                                      reference_positions, 
                                                      target_indices, 
                                                      normalize=normalize,
                                                      transpose_target=self.transpose)

    def _project_positions(self, projection_distance, pair_above, angle_vec, left_position_on_reference, right_position_on_reference):
        pair_rotation_angle = 90 if (pair_above == 1) else -90
        pair_rotation_matrix = np.array([[np.cos(np.deg2rad(pair_rotation_angle)), -np.sin(np.deg2rad(pair_rotation_angle))],
                                        [np.sin(np.deg2rad(pair_rotation_angle)), np.cos(np.deg2rad(pair_rotation_angle))]])
            
        offset_angle_vec = pair_rotation_matrix @ angle_vec
        left_target_position = left_position_on_reference + offset_angle_vec * projection_distance
        right_target_position = right_position_on_reference + offset_angle_vec * projection_distance
        return left_target_position.astype(int), right_target_position.astype(int)

    def _check_point_in_limits(self, point, row_limits, col_limits):
        return (row_limits[0] <= point[0] <= row_limits[1] and
                col_limits[0] <= point[1] <= col_limits[1])

TRIPLET_GENERATORS = {
    'equilateral': EquilateralTripletGenerator,
    'quinn': QuinnTripletGenerator,
}