import itertools
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import typing

from collections import defaultdict


DEFAULT_ORDERS = {
    'model_name': ['mobilenet', 'resnext'],
    'training' : ['saycam(S)', 'random', 'ImageNet']
}
DEFAULT_COLORMAP = plt.get_cmap('Dark2')  # type: ignore
DEFAULT_BAR_KWARGS_BY_FIELD = defaultdict(lambda: defaultdict(dict))
DEFAULT_BAR_KWARGS_BY_FIELD['model_name'] = {name: dict(facecolor=DEFAULT_COLORMAP(i))  # type: ignore
                                             for i, name in enumerate(DEFAULT_ORDERS['model_name'])}
DEFAULT_BAR_KWARGS_BY_FIELD['n_target_types'] = {1: {'hatch': ''}, 2: {'hatch': '/'}}  # type: ignore

DEFAULT_BAR_KWARGS = dict(edgecolor='black')

DEFAULT_TEXT_KWARGS = dict(fontsize=16)

DEFAULT_YLIM = (0, 1.05)


PLOT_PRETTY_NAMES = {
    'above_below': 'Above/Below',
    'n_target_types': 'Target Types',
    'n_habituation_stimuli': 'Habituation Stimuli',
    'mobilenet': 'MobileNetV2',
    'resnext': 'ResNeXt',
    'random': 'Untrained',
    'saycam': 'SAYcam',
    'saycam(S)': 'SAYcam',
    'imagenet': 'ImageNet',
    'Imagenet': 'ImageNet',
    1: 'Same Target',
    2: 'Different Targets',
    's': 'Neither',
    'v': 'Vertical',
    'h': 'Horizontal',
    'hv': 'Both',
    'effective_angle': 'Stimulus Angle',
    'dino': 'DINO',
}

PLOT_PRETTY_NAMES_BY_FIELD = {
    'n_habituation_stimuli': {
        # 1: 'One habituation stimulus',
        # 4: 'Four habituation stimuli',
        1: '1',
        4: '4',
    },
    'triplet_generator': {
        'equidistant': 'Equidistant',
        'diagonal': 'Diagonal',
        'same_half': 'Same Half',
    },
    'same_horizontal_half': {
        True: 'Horizontal Half',
        False: 'Vertical Half',
    },
    'reference_object': {
        'Basket': 'Wicker Basket',
        'CardboardBox': 'Cardboard Box',
        'CardboardBoxNoFlaps': 'Flapless Box',
        'ShortBox': 'Short Box',
        'ShortBoxNoFlaps': 'Flapless Short Box',
        'WoodenBasket': 'Wooden Basket',
    }
}


FIGURE_TEMPLATE = r'''\begin{{figure}}[!htb]
% \vspace{{-0.225in}}
\centering
\includegraphics[width=\linewidth]{{figures/{save_path}}}
\caption{{ {{\bf FIGURE TITLE.}} FIGURE DESCRIPTION.}}
\label{{fig:{label_name}}}
% \vspace{{-0.2in}}
\end{{figure}}
'''
WRAPFIGURE_TEMPLATE = r'''\begin{{wrapfigure}}{{r}}{{0.5\linewidth}}
\vspace{{-.3in}}
\begin{{spacing}}{{1.0}}
\centering
\includegraphics[width=0.95\linewidth]{{figures/{save_path}}}
\caption{{ {{\bf FIGURE TITLE.}} FIGURE DESCRIPTION.}}
\label{{fig:{label_name}}}
\end{{spacing}}
% \vspace{{-.25in}}
\end{{wrapfigure}}'''

SAVE_PATH_PREFIX = 'figures'


def save_plot(save_path, bbox_inches='tight', should_print=False):
    if save_path is not None:
        save_path_no_ext = os.path.splitext(save_path)[0]
        if should_print:
            print('Figure:\n')
            print(FIGURE_TEMPLATE.format(save_path=save_path, label_name=save_path_no_ext.replace('/', '-').replace('_', '-')))
            print('\nWrapfigure:\n')
            print(WRAPFIGURE_TEMPLATE.format(save_path=save_path, label_name=save_path_no_ext.replace('/', '-').replace('_', '-')))
            print('')
        
        if not os.path.isabs(save_path) and not save_path.startswith(SAVE_PATH_PREFIX):
            save_path = os.path.join(SAVE_PATH_PREFIX, save_path)
        
        folder, filename = os.path.split(save_path)
        os.makedirs(folder, exist_ok=True)
        plt.savefig(save_path, bbox_inches=bbox_inches, facecolor=plt.gcf().get_facecolor(), edgecolor='none')


def plot_prettify(text, field_name=None) -> typing.Union[str, typing.List[str]]:
    if isinstance(text, (list, tuple)):
        return [plot_prettify(t) for t in text]  # type: ignore

    if isinstance(text, int) and text > 1000:
        return f'{text // 1000}k ($2^{{ {int(np.log2(text))} }}$)'

    if field_name is not None and field_name in PLOT_PRETTY_NAMES_BY_FIELD:
        if text in PLOT_PRETTY_NAMES_BY_FIELD[field_name]:
            return PLOT_PRETTY_NAMES_BY_FIELD[field_name][text]
    
    if text in PLOT_PRETTY_NAMES:
        return PLOT_PRETTY_NAMES[text]

    text = str(text)
    
    if text in PLOT_PRETTY_NAMES:
        return PLOT_PRETTY_NAMES[text]
    
    for key in PLOT_PRETTY_NAMES:
        if isinstance(key, str) and len(key) > 2 and key in text:
            return PLOT_PRETTY_NAMES[key]

    return text.lower().replace('_', ' ').title()


def filter_and_group(df, filter_dict, group_by_fields, 
                     orders=DEFAULT_ORDERS):
    filtered_df = df.copy(deep=True)
    group_by_fields = group_by_fields[:]
    if 'metric' in group_by_fields:
        group_by_fields.remove('metric')
    
    if filter_dict is not None:
        for filter_name, filter_value in filter_dict.items():
            if isinstance(filter_value, (list, tuple)):
                filtered_df = filtered_df[filtered_df[filter_name].isin(filter_value)]
                if filter_name in orders:
                    orders[filter_name] = list(filter(lambda v: v in filter_value, orders[filter_name]))

            elif filter_value is None:
                filtered_df = filtered_df[filtered_df[filter_name].isnull()]
            
            else:
                filtered_df = filtered_df[filtered_df[filter_name].eq(filter_value)]
            
    return filtered_df.groupby(group_by_fields)


def create_bar_chart(df, filter_dict, group_by_fields, 
                     orders=DEFAULT_ORDERS, title=None,
                     group_bars_by='model', sem=True,
                     above_bar_texts=None, above_bar_text_spacing=0.05,
                     above_bar_text_format='{:.2f}', above_bar_text_fontsize=None,
                     bar_kwargs_by_field=DEFAULT_BAR_KWARGS_BY_FIELD,
                     bar_width=0.2, bar_spacing=0.5, 
                     default_bar_kwargs=DEFAULT_BAR_KWARGS,
                     text_kwargs=DEFAULT_TEXT_KWARGS, title_kwargs=None,
                     add_chance_hline=True, plot_std=True, 
                     ylim=DEFAULT_YLIM, ylabel='Accuracy', 
                     legend_loc='best', legend_ncol=None,
                     save_path=None, save_should_print=False, 
                     ax=None, legend=True, should_show=False):
    
    
    grouped_df = filter_and_group(df, filter_dict, group_by_fields, orders)
    mean = grouped_df.acc_mean.mean()
    if sem:
        std = grouped_df.acc_sem.mean()
    else:
        std = grouped_df.acc_std.mean()
    
    major_group_by = group_by_fields[0]
    minor_group_by = group_by_fields[1:]
    
    if default_bar_kwargs is None:
        default_bar_kwargs = dict()

    if major_group_by in orders:
        major_group_values = orders[major_group_by]
    else:
        major_group_values = mean.index.unique(level=major_group_by)
        
    minor_group_values_list = []
    
    for minor_field_name in minor_group_by:
        if minor_field_name in orders:
            minor_group_values_list.append(orders[minor_field_name])
        else:
            minor_group_values_list.append(mean.index.unique(level=minor_field_name))
    
    major_kwargs = bar_kwargs_by_field[major_group_by]
    minor_kwargs_list = [bar_kwargs_by_field[minor_field_name] for minor_field_name in minor_group_by]
    
    if ax is None:
        figure = plt.figure(figsize=(8, 6))
        ax = plt.gca()
        should_show = True
        
    x = 0
    
    for major_level_value in major_group_values:
        major_level_kwargs = major_kwargs[major_level_value]
        
        for minor_level_value_combination in itertools.product(*minor_group_values_list):
            combined_minor_kwargs = {}
            for i, value in enumerate(minor_level_value_combination):
                combined_minor_kwargs.update(minor_kwargs_list[i][value])
                
            major_and_minor_key = (major_level_value, *minor_level_value_combination)
            
            if major_group_by == 'metric':
                major_and_minor_key = major_and_minor_key[::-1]
                
            m = mean.loc[major_and_minor_key]
            if plot_std:
                s = std.loc[major_and_minor_key]
            else:
                s = None
            
            ax.bar(x, m, yerr=s, width=bar_width, **major_level_kwargs, 
                    **combined_minor_kwargs, **default_bar_kwargs)
            
            if above_bar_texts is not None:
                if above_bar_text_fontsize is None:
                    above_bar_text_fontsize = text_kwargs['fontsize'] - 8
                ax.text(x, m + above_bar_text_spacing, 
                        above_bar_text_format.format(above_bar_texts.loc[major_and_minor_key]),
                        fontsize=above_bar_text_fontsize, fontweight='bold',
                        horizontalalignment='center', verticalalignment='center')

            x += bar_width
        
        x += bar_spacing
        
    minor_group_length = np.product([len(values) for values in minor_group_values_list])
    x_tick_locations = np.arange(len(major_group_values)) * (bar_spacing + bar_width * minor_group_length) +\
                        bar_width * (minor_group_length / 2 - 0.5)
    
    xtick_text_kwargs = text_kwargs.copy()
    if len(major_group_values) > 4:
        xtick_text_kwargs['fontsize'] -= 4
        
    ax.set_xticks(x_tick_locations)
    ax.set_xticklabels([plot_prettify(val) for val in major_group_values], fontdict=xtick_text_kwargs)
    ax.tick_params(axis='both', which='major', labelsize=text_kwargs['fontsize'] - 4)
    
    if add_chance_hline:
        xlim = plt.xlim()
        ax.hlines(0.5, *xlim, linestyle='--', alpha=0.5)
        ax.set_xlim(*xlim)
        
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_xlabel(plot_prettify(major_group_by), **text_kwargs)
    ax.set_ylabel(ylabel, **text_kwargs)
    
    if title_kwargs is None:
        title_kwargs = {}
    if 'fontsize' not in title_kwargs:
        title_kwargs['fontsize'] = text_kwargs['fontsize'] + 4
    
    ax.set_title(title, **title_kwargs)
    
    patches = []
    if legend_ncol:
        ncol = legend_ncol
    else:
        ncol = 0
    for kwarg_set, field_name in zip([major_kwargs] + minor_kwargs_list, group_by_fields):
        if any([len(val) > 0 for val in kwarg_set.values()]):
            if not legend_ncol:
                ncol += 1
                
            for field_value in kwarg_set:
                if field_name in filter_dict and field_value not in filter_dict[field_name]:
                    continue
                
                patch_kwargs = dict(facecolor='none', edgecolor='black')
                patch_kwargs.update(kwarg_set[field_value])
                patches.append(matplotlib.patches.Patch(**patch_kwargs, label=plot_prettify(field_value)))
    
    if len(patches) > 0 and legend: 
        ax.legend(handles=patches, loc=legend_loc, ncol=ncol, fontsize=text_kwargs['fontsize'] - 4)
    
    if save_path is not None:
        save_plot(save_path, should_print=save_should_print)
    
    if should_show:
        plt.show()

