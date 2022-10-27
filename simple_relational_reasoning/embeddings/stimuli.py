from abc import abstractmethod
from functools import lru_cache

import colorcet as cc
import cv2
import matplotlib
import matplotlib.axes
import matplotlib.patches
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas  # type: ignore
import numpy as np
import torch
import torchvision.transforms as transforms
import typing
from torchvision.transforms import functional_tensor
from scipy import ndimage as nd


CACHE_SIZE = 16
DEFAULT_CANVAS_SIZE = (224, 224)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
NORMALIZE = transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
UNNORMALIZE = transforms.Normalize(mean=[-m / s for m, s in zip(NORM_MEAN, NORM_STD)], std=[1.0 /s for s in NORM_STD])


DEFAULT_TARGET_SIZE = 15
DEFAULT_REFERENCE_SIZE = (10, 140)
DEFAULT_COLOR = 'black'
DEFAULT_BLUR_FUNC = lambda x: cv2.blur(x, (5, 5))

def crop_with_fill(img, top: int, left: int, height: int, width: int, fill: int):
    functional_tensor._assert_image_tensor(img)

    w, h = functional_tensor.get_image_size(img)
    right = left + width
    bottom = top + height

    if left < 0 or top < 0 or right > w or bottom > h:
        padding_ltrb = [max(-left, 0), max(-top, 0), max(right - w, 0), max(bottom - h, 0)]
        return functional_tensor.pad(img[..., max(top, 0) : bottom, max(left, 0) : right], padding_ltrb, fill=fill)
    return img[..., top:bottom, left:right]


def build_colored_target_black_reference_stimulus_generator(
    target_size=DEFAULT_TARGET_SIZE, reference_size=DEFAULT_REFERENCE_SIZE, target_colors=['blue', 'green', 'red'], 
    reference_color=DEFAULT_COLOR, blur_func=DEFAULT_BLUR_FUNC, rotate_angle=None, rng=None, **kwargs):

    if kwargs:
        print('Ignoring kwargs: {}'.format(kwargs))

    target_patches = [matplotlib.patches.Circle((0, 0), target_size // 2, color=c) for c in target_colors]
    reference_patch = matplotlib.patches.Ellipse((0, 0), width=reference_size[1], 
                                                 height=reference_size[0], color=reference_color)
                                                    
    return PatchStimulusGenerator(target_size, reference_size, target_patches, 
        reference_patch, rotate_angle=rotate_angle, blur_func=blur_func, rng=rng)


def build_dot_and_bar_stimulus_generator(
    target_size=DEFAULT_TARGET_SIZE, reference_size=DEFAULT_REFERENCE_SIZE, 
    color=DEFAULT_COLOR, rotate_angle=None, rng=None, **kwargs):

    if kwargs:
        print('Ignoring kwargs: {}'.format(kwargs))

    target_patch = matplotlib.patches.Circle((0, 0), target_size // 2, color=color)
    rectangle_reference_patch = matplotlib.patches.Rectangle(
        (-reference_size[1] // 2, -reference_size[0] // 2), reference_size[1], reference_size[0], color=color)
                                                    
    return PatchStimulusGenerator(target_size, reference_size, [target_patch], 
        rectangle_reference_patch, rotate_angle=rotate_angle, rng=rng)


def build_dot_and_ellipse_stimulus_generator(target_size=DEFAULT_TARGET_SIZE, reference_size=DEFAULT_REFERENCE_SIZE, 
    color=DEFAULT_COLOR, blur_func=DEFAULT_BLUR_FUNC, rotate_angle=None, rng=None, **kwargs):

    if kwargs:
        print('Ignoring kwargs: {}'.format(kwargs))

    target_patch = matplotlib.patches.Circle((0, 0), target_size // 2, color=color)
    ellipse_reference_patch = matplotlib.patches.Ellipse((0, 0), width=reference_size[1], 
                                                         height=reference_size[0], color=color)
                                                    
    return PatchStimulusGenerator(target_size, reference_size, [target_patch], 
        ellipse_reference_patch, blur_func=blur_func, rotate_angle=rotate_angle, rng=rng)

def build_differet_shapes_stimulus_generator(target_size=DEFAULT_TARGET_SIZE, reference_size=DEFAULT_REFERENCE_SIZE, 
    color=DEFAULT_COLOR, blur_func=DEFAULT_BLUR_FUNC, rotate_angle=None, rng=None, **kwargs):

    if kwargs:
        print('Ignoring kwargs: {}'.format(kwargs))

    circle_patch = matplotlib.patches.Circle((0, 0), target_size // 2, color=color)
    square_patch = matplotlib.patches.Rectangle(
        # (-target_size // 2, -target_size // 2), 
        (0, -target_size // (2 ** 0.5)), 
        target_size, target_size, angle=45, color=color)
    triangle_patch = matplotlib.patches.RegularPolygon((0, 0), 3, target_size // 2, color=color)
    reference_patch = matplotlib.patches.Ellipse((0, 0), width=reference_size[1], 
                                                height=reference_size[0], color=color)

    return PatchStimulusGenerator(target_size, reference_size, 
        [circle_patch, square_patch, triangle_patch], reference_patch, 
        rotate_angle=rotate_angle, blur_func=blur_func, rng=rng)


def build_split_text_stimulus_generator(target_size=DEFAULT_TARGET_SIZE, 
    reference_box_size=10, total_reference_size=(10, 140), n_reference_patches=8,
    color=DEFAULT_COLOR, reference_patch_kwargs=None, rotate_angle=None, rng=None, **kwargs):

    if kwargs:
        print('Ignoring kwargs: {}'.format(kwargs))

    if reference_patch_kwargs is None:
        reference_patch_kwargs = {}

    triangle_patch = matplotlib.patches.RegularPolygon((0, 0), 3, target_size // 2, color=color)
    reference_patches = [matplotlib.patches.Rectangle(((-reference_box_size // 2) + (reference_box_size * 2 * i) - (reference_box_size * (n_reference_patches - 1)), 
                                                   (-reference_box_size // 2)), 
                                                  reference_box_size, reference_box_size, color=color)
                         for i in range(n_reference_patches)]

    return PatchStimulusGenerator(target_size, total_reference_size, 
                                  ['E', '$+$', triangle_patch, 's', '$\\to$'], 
                                  reference_patches, rotate_angle=rotate_angle,
                                  reference_patch_kwargs=reference_patch_kwargs, rng=rng)


def build_random_color_stimulus_generator(rng, cmap=cc.cm.glasbey,
    target_size=DEFAULT_TARGET_SIZE, reference_size=DEFAULT_REFERENCE_SIZE, 
    blur_func=DEFAULT_BLUR_FUNC, rotate_angle=None, **kwargs):

    if kwargs:
        print('Ignoring kwargs: {}'.format(kwargs))

    def target_patch_func(color_index):
        return matplotlib.patches.Circle((0, 0), target_size // 2, color=cmap(color_index))

    def reference_patch_func(color_index):
        return matplotlib.patches.Ellipse((0, 0), width=reference_size[1], 
                                                 height=reference_size[0], 
                                                 color=cmap(color_index))

    return PatchStimulusGenerator(target_size, reference_size, target_patch_func,
        reference_patch_func, blur_func=blur_func, rotate_angle=rotate_angle, rng=rng)


STIMULUS_GENERATORS = {
    'colored_dot_black_ellipse': build_colored_target_black_reference_stimulus_generator,
    'black_dot_and_bar': build_dot_and_bar_stimulus_generator,
    'same_color_dot_and_ellipse': build_dot_and_ellipse_stimulus_generator,
    'different_shapes': build_differet_shapes_stimulus_generator,
    'split_text': build_split_text_stimulus_generator,
    'random_color': build_random_color_stimulus_generator,
}


DEFAULT_MIN_ROTATE_MARGIN = 5
DEFAULT_CENTROID_PATCH_SIZE = 2
DEFAULT_ROTATE_PADDING = 100
DEFAULT_MARGIN_BUFFER = 2

class StimulusGenerator:
    target_size: typing.Tuple[int, int]
    reference_size: typing.Tuple[int, int]
    canvas_size: typing.Tuple[int, int]

    def __init__(self, target_size, reference_size, canvas_size, rotate_angle=None, 
        min_rotate_margin=DEFAULT_MIN_ROTATE_MARGIN, 
        centroid_patch_size=DEFAULT_CENTROID_PATCH_SIZE, centroid_marker_value=0, 
        padding=DEFAULT_ROTATE_PADDING, margin_buffer=DEFAULT_MARGIN_BUFFER, dtype=torch.float32, rng=None):

        self.target_size = target_size
        self.reference_size = reference_size
        self.canvas_size = canvas_size
        self.rotate_angle = rotate_angle
        self.min_rotate_margin = min_rotate_margin
        self.centroid_patch_size = centroid_patch_size
        self.centroid_marker_value = centroid_marker_value
        self.centroid_marker_value_tensor = torch.tensor(self.centroid_marker_value, dtype=dtype)
        self.padding = padding
        self.margin_buffer = margin_buffer
        self.dtype = dtype
        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng
        self.n_target_types = 1
    
    def generate(self, target_position, reference_positions, target_index=0, 
                 center_positions=True, transpose_target=False, pad_and_crop=True,
                 multiple_target_positions=False, target_colors=None) -> torch.Tensor:
        if len(reference_positions) > 0 and not hasattr(reference_positions[0], '__len__'):
            reference_positions = [reference_positions]
        
        x = self._canvas(padding=self.padding if pad_and_crop else 0)
        if pad_and_crop:
            target_position = [np.array(t) + self.padding for t in target_position]
            reference_positions = [[r + self.padding for r in rp] for rp in reference_positions]
        
        # reference first in case of overlap
        reference_centering = np.array([center_positions * s // 2 for s in self.reference_size])
        for ref_pos in reference_positions:
            rp = np.array(ref_pos) - reference_centering
            
#             if torch.any(torch.eq(torch.tensor(x[:, rp[0]:rp[0] + self.reference_size[0],
#                  rp[1]:rp[1] + self.reference_size[1]].shape), 0)):
#                 print(ref_pos, rp)
            ref_object = self._reference_object()

            x[:, rp[0]:rp[0] + ref_object.shape[1],
                 rp[1]:rp[1] + ref_object.shape[2]] = ref_object
            
        target_centering = np.array([center_positions * s // 2 for s in self.target_size])

        if multiple_target_positions:
            for i, t_pos in enumerate(target_position):
                target_pos = np.array(t_pos) - target_centering
                target = self._target_object(target_index)
                if transpose_target:
                    target = np.transpose(target, (0, 2, 1))
                
                if target_colors is not None:
                    target = torch.clone(target)
                    target[:, (target != 1).all(0)] = torch.tensor(target_colors[i], dtype=target.dtype).view(-1, 1)
                
                x[:, target_pos[0]:target_pos[0] + target.shape[1],
                    target_pos[1]:target_pos[1] + target.shape[2]] = target

        else:

            target_pos = np.array(target_position) - target_centering
            target = self._target_object(target_index)
            if transpose_target:
                target = np.transpose(target, (0, 2, 1))
            
            # if (target_pos < 0).any() or ((target_pos + np.array(target.shape[1:])) > canvas_shape[0]).any():
            #     print(f'Target out of bounds: target: {target_pos}, centering: {target_centering}, references: {reference_positions}, angle: {self.rotate_angle} ')

            x[:, target_pos[0]:target_pos[0] + target.shape[1],
                target_pos[1]:target_pos[1] + target.shape[2]] = target

        if self.rotate_angle is not None and self.rotate_angle != 0:
            if pad_and_crop is False:
                raise NotImplementedError('Cannot rotate without padding and cropping')
            x = transforms.functional.rotate(x, self.rotate_angle, fill=[1.0, 1.0, 1.0])  # type: ignore

        return x
    
    def __call__(self, target_position, reference_positions, transpose_target=False) -> torch.Tensor:
        return NORMALIZE(self.generate(target_position, reference_positions, 
            transpose_target=transpose_target, pad_and_crop=False))
        
    def batch_generate(self, target_positions, reference_positions, target_indices=None, 
                       normalize=True, transpose_target=False, pad_and_crop=True, return_centroid=False, crop_to_center=False) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, np.ndarray]]:
        
        if reference_positions is None:
            reference_positions = []
        
        if len(reference_positions) == 0:
            reference_positions = [tuple()] * len(target_positions)

        else:
            if len(reference_positions) != len(target_positions):
                if isinstance(reference_positions[0], np.ndarray):
                    reference_positions = [tuple([tuple(ref) for ref in reference_positions])] * len(target_positions)
                elif hasattr(reference_positions[0], '__len__'):
                    reference_positions = [tuple(reference_positions)] * len(target_positions)
                else:
                    reference_positions = [(tuple(reference_positions),)] * len(target_positions)
            
        if target_indices is None:
            target_indices = (0,) * len(target_positions)
        elif isinstance(target_indices, int):
            target_indices = (target_indices,) * len(target_positions)
        elif not isinstance(target_indices, tuple):
            target_indices = tuple(target_indices)
            
        if len(target_indices) != len(target_positions):
            raise ValueError(f'Expected target indices (N={len(target_indices)} to have the same length as the target positions (N={len(target_positions)}. Aborting...')
            
        target_positions = tuple([tuple(pos) for pos in target_positions])
        reference_positions = tuple(reference_positions)
        stimulus, centroid = self.cached_batch_generate(target_positions, reference_positions, target_indices, 
                                          normalize=normalize, transpose_target=transpose_target, 
                                          pad_and_crop=pad_and_crop, crop_to_center=crop_to_center)

        if return_centroid:
            return stimulus, centroid

        return stimulus
                
    @lru_cache(maxsize=CACHE_SIZE)
    def cached_batch_generate(self, target_positions, reference_positions, target_indices, 
                              normalize=True, transpose_target=False, pad_and_crop=True, 
                              crop_to_center=False, multiple_target_positions=False, target_colors=None) -> typing.Tuple[torch.Tensor, np.ndarray]:
        
        self.new_stimulus()
        zip_gen = zip(target_positions, reference_positions, target_indices)

        stimuli = torch.stack(
            [self.generate(tp, rp, i, 
                          transpose_target=transpose_target, 
                          pad_and_crop=pad_and_crop, 
                          multiple_target_positions=multiple_target_positions,
                          target_colors=target_colors) 
             for (tp, rp, i) in zip_gen]
        )

        centroid = None
        
        if pad_and_crop:
            first_non_empty_row, last_non_empty_row, first_non_empty_col, last_non_empty_col = \
                find_non_empty_indices(stimuli, empty_value=EMPTY_TENSOR_PIXEL.view(1, 3, 1, 1), color_axis=1)

            stimuli = stimuli[:, :, first_non_empty_row:last_non_empty_row, first_non_empty_col:last_non_empty_col]

            n_rows, n_cols = stimuli.shape[2:]
            if crop_to_center:
                top = self.canvas_size[0] // 2 - n_rows // 2
                left = self.canvas_size[1] // 2 - n_cols // 2
                centroid = np.array([self.canvas_size[0] // 2, self.canvas_size[1] // 2], dtype=np.int)
                
            else:
                top = self.rng.integers(self.margin_buffer, self.canvas_size[0] - n_rows - self.margin_buffer)
                left = self.rng.integers(self.margin_buffer, self.canvas_size[1] - n_cols - self.margin_buffer)
                centroid = np.array([top + (n_rows // 2), left + (n_cols // 2)], dtype=np.int)
                
            new_canvas = self._canvas(n=stimuli.shape[0])
            new_canvas[:, :, top:top + n_rows, left:left + n_cols] = stimuli
            stimuli = new_canvas

        if normalize:
            return NORMALIZE(stimuli), centroid  # type: ignore
        
        return stimuli, centroid  # type: ignore

    def _to_tensor(self, t):
        return torch.tensor(t, dtype=self.dtype)
    
    def _validate_input_to_tuple(self, input_value, n_args=2):
        if not hasattr(input_value, '__len__') or len(input_value) == 1:
            return (input_value, ) * n_args
        
        return input_value
    
    def _validate_color_input(self, c):
        if isinstance(c, str):
            t = self._to_tensor(matplotlib.colors.to_rgb(c))
        else:
            t = self._to_tensor(self._validate_input_to_tuple(c, 3))
        
        return t.view(3, 1, 1)
    
    @abstractmethod
    def _canvas(self, n=None, padding=0) -> torch.Tensor:
        pass
    
    @abstractmethod
    def _reference_object(self) -> torch.Tensor:
        pass
    
    @abstractmethod
    def _target_object(self, index=0) -> torch.Tensor:
        pass
    
    def new_stimulus(self) -> None:
        pass

class NaiveStimulusGenerator(StimulusGenerator):
    def __init__(self, target_size, reference_size, canvas_size=DEFAULT_CANVAS_SIZE,
                 target_color='black', reference_color='blue', background_color='white',
                 rotate_angle=None, dtype=torch.float32):
        super(NaiveStimulusGenerator, self).__init__(target_size, reference_size, canvas_size, rotate_angle=rotate_angle, dtype=dtype)
        
        self.target_size = self._validate_input_to_tuple(target_size)
        self.reference_size = self._validate_input_to_tuple(reference_size)
        self.canvas_size = self._validate_input_to_tuple(canvas_size)
        
        self.target_color = self._validate_color_input(target_color)
        self.reference_color = self._validate_color_input(reference_color)
        self.background_color = self._validate_color_input(background_color)
        
    def _canvas(self, n=None, padding=0) -> torch.Tensor:
        canvas_size = (self.canvas_size[0] + (2 * padding), self.canvas_size[1] + (2 * padding))

        if n is None:
            return torch.ones(3, *canvas_size, dtype=self.dtype) * self.background_color

        else:
            return torch.ones(n, 3, *canvas_size, dtype=self.dtype) * self.background_color.view(1, 3, 1, 1)
    
    def _reference_object(self) -> torch.Tensor:
        return self.reference_color
    
    def _target_object(self, index=0) -> torch.Tensor:
        return self.target_color



EMPTY_PIXEL = np.array([255, 255, 255], dtype=np.uint8)
EMPTY_TENSOR_PIXEL = torch.tensor([1., 1., 1.], dtype=torch.float32).view(3, 1, 1)

def find_non_empty_indices(X, empty_value=EMPTY_PIXEL, color_axis=2):
    if isinstance(X, np.ndarray):
        if not isinstance(empty_value, np.ndarray):
            raise ValueError('Expected empty_value to be a numpy array when X is a numpy array')

        empty_pixels = (X[:, :, :-1] == empty_value).all(axis=color_axis)
        non_empty_rows = ~(empty_pixels.all(axis=1))
        non_empty_cols = ~(empty_pixels.all(axis=0))
        
        first_non_empty_row = non_empty_rows.argmax()
        last_non_empty_row = non_empty_rows.shape[0] - non_empty_rows[::-1].argmax()

        first_non_empty_col = non_empty_cols.argmax() 
        last_non_empty_col = non_empty_cols.shape[0] - non_empty_cols[::-1].argmax()

    elif isinstance(X, torch.Tensor):
        if not isinstance(empty_value, torch.Tensor):
            raise ValueError('Expected empty_value to be a torch tensor when X is a torch tensor')

        empty_pixels = (X == empty_value).all(dim=color_axis)
        if empty_pixels.dim() == 3:
            empty_pixels = empty_pixels.all(dim=0)
        non_empty_rows = (~(empty_pixels.all(dim=1))).double()  # torch doesn't support argmax for booleans
        non_empty_cols = (~(empty_pixels.all(dim=0))).double()
        
        first_non_empty_row = non_empty_rows.argmax()
        last_non_empty_row = non_empty_rows.shape[0] - non_empty_rows.flip(0).argmax()

        first_non_empty_col = non_empty_cols.argmax() 
        last_non_empty_col = non_empty_cols.shape[0] - non_empty_cols.flip(0).argmax()

    else:
        raise ValueError('Expected X to be a numpy array or a torch tensor')


    return first_non_empty_row, last_non_empty_row, first_non_empty_col, last_non_empty_col


class PatchStimulusGenerator(StimulusGenerator):
    def _patch_to_array(self, patch, size, xlim=None, ylim=None, fontsize=16):
        fig = Figure(figsize=(4, 4))
        # attach a non-interactive Agg canvas to the figure
        # (as a side-effect of the ``__init__``)
        canvas = FigureCanvas(fig)
        ax = typing.cast(matplotlib.axes.Axes, fig.subplots())
        max_size = max(size)

        if xlim is None:
            ax.set_xlim(-max_size, max_size)
        else:
            ax.set_xlim(*xlim)

        if ylim is None:
            ax.set_ylim(-max_size, max_size)
        else:
            ax.set_ylim(*ylim)

        ax.set_facecolor(np.array(self.background_color).squeeze())
        
        if isinstance(patch, (list, tuple)):
            for p in patch:
                if isinstance(p, str):
                    ax.text(0, 0, p, fontsize=fontsize)
                else:
                    ax.add_patch(p)
        else:
            if isinstance(patch, str):
                ax.text(0, 0, patch, fontsize=fontsize)
            else:
                ax.add_patch(patch)  
            
        ax.set_axis_off()
        # ax.autoscale(tight=True)
            
        # Force a draw so we can grab the pixel buffer
        canvas.draw()
        # grab the pixel buffer and dump it into a numpy array
        X = np.array(canvas.renderer.buffer_rgba())

        # if self.blur_func is not None:
        #     X = self.blur_func(X)
        #     if not np.issubdtype(X.dtype, np.int):
        #         X = (X * 255).astype(np.uint8)
        
        # print(X.shape, X.dtype, X[0, 0], size)
        # plt.imshow(X)
        # plt.show()

        X_resized = self.trim_and_resize(X, size)

        # print(X_resized.shape, X_resized.dtype, X_resized[0, 0], size)
        # plt.imshow(X_resized)
        # plt.show()

        X_rgb = cv2.cvtColor(X_resized, cv2.COLOR_RGBA2RGB)

        if self.blur_func is not None:
            X_rgb = self.blur_func(X_rgb)
        
        X_float_tensor = torch.tensor(X_rgb, dtype=self.dtype).permute(2, 0, 1)
        return X_float_tensor / X_float_tensor.max()

    def trim_and_resize(self, X, size):
        first_non_empty_row, last_non_empty_row, first_non_empty_col, last_non_empty_col = find_non_empty_indices(X)
        # print(row_start, row_end, col_start, col_end)
        X_trim = X[first_non_empty_row:last_non_empty_row, first_non_empty_col:last_non_empty_col, :]
        # plt.imshow(X_trim)
        # plt.show()
        X_resized = cv2.resize(X_trim, dsize=size[::-1])
        # plt.imshow(X_resized)
        # plt.show()

        return X_resized
    
    def __init__(self, target_size, reference_size, target_patch, reference_patch,
                 blur_func=None, target_patch_kawrgs=None, reference_patch_kwargs=None,
                 canvas_size=DEFAULT_CANVAS_SIZE, rotate_angle=None,
                 background_color='white', rng=None, cmap_max_color=256, function_n_target_types=5, dtype=torch.float32):
        super(PatchStimulusGenerator, self).__init__(target_size, reference_size, canvas_size, dtype=dtype, rng=rng)

        if target_patch_kawrgs is None:
            target_patch_kawrgs = {}
            
        if reference_patch_kwargs is None:
            reference_patch_kwargs = {}
        
        self.target_size = self._validate_input_to_tuple(target_size)
        self.reference_size = self._validate_input_to_tuple(reference_size)
        self.canvas_size = self._validate_input_to_tuple(canvas_size)
        
        self.rotate_angle = rotate_angle

        self.blur_func = blur_func
        self.background_color = self._validate_color_input(background_color)
        
        self.random_max_color = cmap_max_color - function_n_target_types
        self.patch_rng_index = 0
        
        if callable(target_patch):
            self.target_patch_func = target_patch
            self.target_patch_kawrgs = target_patch_kawrgs   
            self.n_target_types = function_n_target_types
            self.target_patch_cache = {}

        else:
            self.target_patch_func = None
            if not isinstance(target_patch, (list, tuple)):
                target_patch = [target_patch]
            
            self.targets_arrs = [self._patch_to_array(patch, self.target_size, **target_patch_kawrgs) for patch in target_patch]
            self.n_target_types = len(self.targets_arrs)

        if callable(reference_patch):
            self.reference_patch_func = reference_patch
            self.reference_patch_kwargs = reference_patch_kwargs
            self.reference_patch_cache = {}
            
        else:
            self.reference_patch_func = None
            self.reference_arr = self._patch_to_array(reference_patch, self.reference_size, **reference_patch_kwargs)

    def new_stimulus(self):
        if self.rng is not None:
            self.patch_rng_index = self.rng.integers(self.random_max_color, size=1)[0]

    def _canvas(self, n=None, padding=0):
        canvas_size = (self.canvas_size[0] + (2 * padding), self.canvas_size[1] + (2 * padding))

        if n is None:
            return torch.ones(3, *canvas_size, dtype=self.dtype) * self.background_color

        else:
            return torch.ones(n, 3, *canvas_size, dtype=self.dtype) * self.background_color.view(1, 3, 1, 1)
    
    def _reference_object(self):
        if self.reference_patch_func is None:
            return self.reference_arr

        return self._cached_reference_object(self.patch_rng_index)

    @lru_cache(maxsize=256)
    def _cached_reference_object(self, index):
        reference_patch = self.reference_patch_func(index)  # type: ignore
        return self._patch_to_array(reference_patch, self.reference_size, **self.reference_patch_kwargs)
        
    def _target_object(self, index=0):
        if self.target_patch_func is None:
            return self.targets_arrs[index]

        return self._cached_target_object(self.patch_rng_index + index + 1)

    @lru_cache(maxsize=256)
    def _cached_target_object(self, index):
        target_patch = self.target_patch_func(index)  # type: ignore
        return self._patch_to_array(target_patch, self.target_size, **self.target_patch_kawrgs)
