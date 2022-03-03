from abc import abstractmethod
from functools import lru_cache
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas

CACHE_SIZE = 16
DEFAULT_CANVAS_SIZE = (224, 224)
NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

DEFAULT_TARGET_SIZE = 15
DEFAULT_REFERENCE_SIZE = (10, 100)
DEFAULT_COLOR = 'black'
DEFAULT_BLUR_FUNC = lambda x: cv2.blur(x, (5, 5))


def build_colored_target_black_reference_stimulus_generator(
    target_size=DEFAULT_TARGET_SIZE, reference_size=DEFAULT_REFERENCE_SIZE, target_colors=['blue', 'green', 'red'], 
    reference_color=DEFAULT_COLOR, blur_func=DEFAULT_BLUR_FUNC, **kwargs):

    if kwargs:
        print('Ignoring kwargs: {}'.format(kwargs))

    target_patches = [matplotlib.patches.Circle((0, 0), target_size // 2, color=c) for c in target_colors]
    reference_patch = matplotlib.patches.Ellipse((0, 0), width=reference_size[1], 
                                                 height=reference_size[0], color=reference_color)
                                                    
    return PatchStimulusGenerator(target_size, reference_size, target_patches, reference_patch, blur_func=blur_func)


def build_dot_and_bar_stimulus_generator(
    target_size=DEFAULT_TARGET_SIZE, reference_size=DEFAULT_REFERENCE_SIZE, 
    color=DEFAULT_COLOR, **kwargs):

    if kwargs:
        print('Ignoring kwargs: {}'.format(kwargs))

    target_patch = matplotlib.patches.Circle((0, 0), target_size // 2, color=color)
    rectangle_reference_patch = matplotlib.patches.Rectangle(
        (-reference_size[1] // 2, -reference_size[0] // 2), reference_size[1], reference_size[0], color=color)
                                                    
    return PatchStimulusGenerator(target_size, reference_size, [target_patch], rectangle_reference_patch)


def build_dot_and_ellipse_stimulus_generator(target_size=DEFAULT_TARGET_SIZE, reference_size=DEFAULT_REFERENCE_SIZE, 
    color=DEFAULT_COLOR, blur_func=DEFAULT_BLUR_FUNC, **kwargs):

    if kwargs:
        print('Ignoring kwargs: {}'.format(kwargs))

    target_patch = matplotlib.patches.Circle((0, 0), target_size // 2, color=color)
    ellipse_reference_patch = matplotlib.patches.Ellipse((0, 0), width=reference_size[1], 
                                                         height=reference_size[0], color=color)
                                                    
    return PatchStimulusGenerator(target_size, reference_size, [target_patch], 
        ellipse_reference_patch, blur_func=blur_func)

def build_differet_shapes_stimulus_generator(target_size=DEFAULT_TARGET_SIZE, reference_size=DEFAULT_REFERENCE_SIZE, 
    color=DEFAULT_COLOR, blur_func=DEFAULT_BLUR_FUNC, **kwargs):

    if kwargs:
        print('Ignoring kwargs: {}'.format(kwargs))

    circle_patch = matplotlib.patches.Circle((0, 0), target_size // 2, color=color)
    square_patch = matplotlib.patches.Rectangle((-target_size // 2, -target_size // 2), target_size, target_size, color=color)
    triangle_patch = matplotlib.patches.RegularPolygon((0, 0), 3, target_size // 2, color=color)
    reference_patch = matplotlib.patches.Ellipse((0, 0), width=reference_size[1], 
                                                height=reference_size[0], color=color)

    return PatchStimulusGenerator(target_size, reference_size, 
        [circle_patch, square_patch, triangle_patch], reference_patch, blur_func=blur_func)


def build_split_text_stimulus_generator(target_size=DEFAULT_TARGET_SIZE, 
    reference_box_size=8, total_reference_size=(10, 140), n_reference_patches=7,
    color=DEFAULT_COLOR, reference_patch_kwargs=None, **kwargs):

    if kwargs:
        print('Ignoring kwargs: {}'.format(kwargs))

    if reference_patch_kwargs is None:
        reference_patch_kwargs = {}

    triangle_patch = matplotlib.patches.RegularPolygon((0, 0), 3, target_size[0] // 2, color=color)
    reference_patches = [matplotlib.patches.Rectangle(((-reference_box_size // 2) + (reference_box_size * 2 * i), 
                                                   (-reference_box_size // 2)), 
                                                  reference_box_size, reference_box_size, color=color)
                         for i in range(n_reference_patches)]

    return PatchStimulusGenerator(target_size, total_reference_size, 
                                  ['E', '$+$', triangle_patch, 's', '$\\to$'], 
                                  reference_patches, 
                                  reference_patch_kwargs=reference_patch_kwargs)

STIMULUS_GENERATORS = {
    'colored_dot_black_ellipse': build_colored_target_black_reference_stimulus_generator,
    'black_dot_and_bar': build_dot_and_bar_stimulus_generator,
    'same_color_dot_and_ellipse': build_dot_and_ellipse_stimulus_generator,
    'different_shapes': build_differet_shapes_stimulus_generator,
    'split_text': build_split_text_stimulus_generator,
}



class StimulusGenerator:
    def __init__(self, target_size, reference_size, rotate_angle=None, dtype=torch.float32):
        self.target_size = target_size
        self.reference_size = reference_size
        self.rotate_angle = rotate_angle
        self.dtype = dtype
        
        self.n_target_types = 1
    
    def generate(self, target_position, reference_positions, target_index=0, 
                 center_positions=True, transpose_target=False, stimulus_centroid=None) -> torch.Tensor:
        if not hasattr(reference_positions[0], '__len__'):
            reference_positions = [reference_positions]
        
        x = self._canvas()
        
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
        target_pos = np.array(target_position) - target_centering
        target = self._target_object(target_index)
        if transpose_target:
            target = np.transpose(target, (0, 2, 1))
            
        x[:, target_pos[0]:target_pos[0] + target.shape[1],
             target_pos[1]:target_pos[1] + target.shape[2]] = target

        if self.rotate_angle is not None:
            if stimulus_centroid is None:
                stimulus_centroid = np.array([s // 2 for s in x.shape[1:]], dtype=np.int)

            # record current value at the centroid, mark it, find it again after rotating, crop such that it's centered
            original_centroid_value = x[:, stimulus_centroid[0], stimulus_centroid[1]]
            centroid_marker = None

            while centroid_marker is None:
                centroid_marker = torch.randint(0, 101, (3, 1, 1), dtype=self.dtype) / 100
                marker_exists = (x == centroid_marker).all(axis=0).any().item()
                if marker_exists:
                    centroid_marker = None

            x[:, stimulus_centroid[0], stimulus_centroid[1]] = centroid_marker.squeeze()

            x_rot = transforms.functional.rotate(x, self.rotate_angle, center=tuple(stimulus_centroid),
                expand=True, fill=[1.0, 1.0, 1.0])
            
            if x_rot.shape != x.shape:
                new_centroid = (x_rot == centroid_marker).all(axis=0).nonzero().squeeze().numpy()
                if len(new_centroid.shape) > 1:
                    new_centroid = new_centroid[0]

                # TODO: check if this would override bounds, and if it does, min/max it
                top, left = new_centroid - stimulus_centroid
                top = np.clip(top, 0, x_rot.shape[1] - x.shape[1])
                left = np.clip(left, 0, x_rot.shape[2] - x.shape[2])

                x_rot = transforms.functional.crop(x_rot, top, left, *x.shape[1:])
                x_rot[:, stimulus_centroid[0], stimulus_centroid[1]] = original_centroid_value

            x = x_rot
        
        return x
    
    def __call__(self, target_position, reference_positions, transpose_target=False, stimulus_centroid=None) -> torch.Tensor:
        return NORMALIZE(self.generate(target_position, reference_positions, 
            transpose_target=transpose_target, stimulus_centroid=stimulus_centroid))
        
    def batch_generate(self, target_positions, reference_positions, target_indices=None, 
                       normalize=True, transpose_target=False, stimulus_centroid=None) -> torch.Tensor:
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
        if stimulus_centroid is not None:
            stimulus_centroid = tuple(stimulus_centroid)
        return self.cached_batch_generate(target_positions, reference_positions, target_indices, 
                                          normalize=normalize, transpose_target=transpose_target, 
                                          stimulus_centroid=stimulus_centroid)
                
    @lru_cache(maxsize=CACHE_SIZE)
    def cached_batch_generate(self, target_positions, reference_positions, target_indices, 
                              normalize=True, transpose_target=False, stimulus_centroid=None):
        zip_gen = zip(target_positions, reference_positions, target_indices)
        if normalize:
            return torch.stack([NORMALIZE(self.generate(t, p, i, transpose_target=transpose_target, stimulus_centroid=stimulus_centroid)) 
                                for (t, p, i) in zip_gen])
        else:
            return torch.stack([self.generate(t, p, i, transpose_target=transpose_target, stimulus_centroid=stimulus_centroid) 
                                for (t, p, i) in zip_gen])

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
            t = self.to_tensor(self._validate_input_to_tuple(c, 3))
        
        return t.view(3, 1, 1)
    
    @abstractmethod
    def _canvas(self):
        pass
    
    @abstractmethod
    def _reference_object(self):
        pass
    
    @abstractmethod
    def _target_object(self, index=0):
        pass
    

class NaiveStimulusGenerator(StimulusGenerator):
    def __init__(self, target_size, reference_size, canvas_size=DEFAULT_CANVAS_SIZE,
                 target_color='black', reference_color='blue', background_color='white',
                 rotate_angle=None, dtype=torch.float32):
        super(NaiveStimulusGenerator, self).__init__(target_size, reference_size, rotate_angle=rotate_angle, dtype=dtype)
        
        self.target_size = self._validate_input_to_tuple(target_size)
        self.reference_size = self._validate_input_to_tuple(reference_size)
        self.canvas_size = self._validate_input_to_tuple(canvas_size)
        
        self.target_color = self._validate_color_input(target_color)
        self.reference_color = self._validate_color_input(reference_color)
        self.background_color = self._validate_color_input(background_color)
        
    def _canvas(self):
        return torch.ones(3, *self.canvas_size, dtype=self.dtype) * self.background_color
    
    def _reference_object(self):
        return self.reference_color
    
    def _target_object(self, index=0):
        return self.target_color
    

EMPTY_PIXEL = np.array([255, 255, 255, 0], dtype=np.uint8)

class PatchStimulusGenerator(StimulusGenerator):
    def _patch_to_array(self, patch, size, xlim=None, ylim=None, fontsize=16):
        fig = Figure(figsize=(4, 4))
        # attach a non-interactive Agg canvas to the figure
        # (as a side-effect of the ``__init__``)
        canvas = FigureCanvas(fig)
        ax = fig.subplots()
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
        
        # print(X.shape, X.dtype, X[0, 0], size)
        # plt.imshow(X)
        # plt.show()

        X_resized = self.trim_and_resize(X, size)
        X_rgb = cv2.cvtColor(X_resized, cv2.COLOR_RGBA2RGB)
        if self.blur_func is not None:
            X_rgb = self.blur_func(X_rgb)
        X_float_tensor = torch.tensor(X_rgb, dtype=self.dtype).permute(2, 0, 1)
        return X_float_tensor / X_float_tensor.max()

    def trim_and_resize(self, X, size):
        row_start = 0
        for i in range(X.shape[0]):
            if not np.all(X[i] == EMPTY_PIXEL):
                row_start = i
                break

        row_end = X.shape[0]
        for i in range(X.shape[0] - 1, 0, -1):
            if not np.all(X[i] == EMPTY_PIXEL):
                row_end = i
                break

        col_start = 0
        for i in range(X.shape[1]):
            if not np.all(X[:,i] == EMPTY_PIXEL):
                col_start = i
                break

        col_end = X.shape[1]
        for i in range(X.shape[1] - 1, 0, -1):
            if not np.all(X[:,i] == EMPTY_PIXEL):
                col_end = i
                break
                
        # print(row_start, row_end, col_start, col_end)
        X_trim = X[row_start:row_end + 1, col_start:col_end + 1, :]
        # plt.imshow(X_trim)
        # plt.show()
        X_resized = cv2.resize(X_trim, dsize=size[::-1])
        # plt.imshow(X_resized)
        # plt.show()

        return X_resized
    
    def __init__(self, target_size, reference_size, target_patch, reference_patch,
                 blur_func=None, target_patch_kawrgs=None, reference_patch_kwargs=None,
                 canvas_size=DEFAULT_CANVAS_SIZE, rotate_angle=None,
                 background_color='white', dtype=torch.float32):
        super(PatchStimulusGenerator, self).__init__(target_size, reference_size, dtype)
        
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
        
        if not isinstance(target_patch, (list, tuple)):
            target_patch = [target_patch]
          
        self.targets_arrs = [self._patch_to_array(patch, self.target_size, **target_patch_kawrgs) for patch in target_patch]
        self.n_target_types = len(self.targets_arrs)
        self.reference_arr = self._patch_to_array(reference_patch, self.reference_size, **reference_patch_kwargs)
        
    def _canvas(self):
        return torch.ones(3, *self.canvas_size, dtype=self.dtype) * self.background_color
    
    def _reference_object(self):
        return self.reference_arr
    
    def _target_object(self, index=0):
        return self.targets_arrs[index]
