import itertools
import numpy as np
import pandas as pd
import tabulate

from collections import defaultdict
from IPython.display import display, Markdown


MODEL_ORDERS = [f'{model}-{dataset}' for (dataset, model) 
                in itertools.product(('saycam(S)', 'imagenet', 'random'), ('mobilenet', 'resnext'))]

HEADERS = ['Stimulus Rendering', 'Target Type'] + MODEL_ORDERS
DF_HEADERS = ['model_name', 'condition', 'acc_mean', 'acc_std', 'acc_sem'] 

PRETTY_NAMES = {
    'Quinn-Split-Reference-Text': 'Quinn-like',
    'Quinn-Diff-Shapes': 'Geometric Shapes',
    'Quinn-Random-Color': 'Colors'
}

SIMPLE_NAMES = {
    'Quinn-Split-Reference-Text': 'quinn',
    'Quinn-like': 'quinn',
    'Quinn-Diff-Shapes': 'shapes',
    'Geometric Shapes': 'shapes',
    'Quinn-Random-Color': 'colors',
    'Colors': 'colors',
    'Above/Below': 'above_below',
    'Between': 'between',
    'Left/Right': 'left_right',
    'VerticalBetween': 'vertical_between',
}


def prettify(text):
    if text in PRETTY_NAMES:
        return PRETTY_NAMES[text]
    
    return text


def simplify(text):
    if text in SIMPLE_NAMES:
        return SIMPLE_NAMES[text]
    
    return text


def parse_condition_name(name, clean_func=prettify):
    name = name.replace('Between-sideways', 'VerticalBetween')
    n_types = int(name[-7])
    name_without_types = name[:-8]
    condition, relation = name_without_types.rsplit('-', 1)
        
    return prettify(relation), prettify(condition), n_types


def format_results(task_results, print_std=True):
    if print_std:
        return f'${task_results.mean:.4f} \\pm {task_results.std / (task_results.n ** 0.5):.4f}$'
    else:
        return f'${task_results.mean:.4f}$'


def format_replication_results(task_result_lists, print_std=True):
    result_means = np.array([tr.mean for tr in task_result_lists])
    if print_std:
        return f'${result_means.mean():.4f} \\pm {result_means.std() / (len(result_means) ** 0.5):.4f}$'
    else:
        return f'${result_means.mean():.4f}$'


def display_multiple_model_results_single_type(group_name, results, tablefmt='github'):
    display(Markdown(f'## {group_name} Results'))
        
    metric_names = tuple(next(iter(next(iter(results.values())).values())).keys())
    headers = ['Model'] + [name + '&nbsp; ' * 12 for name in metric_names]
    
    for key in next(iter(results.values())).keys():
        display(Markdown(f'### {key}'))
        rows = []
        for model_name, model_results in results.items():
            rows.append([model_name] + [format_results(model_results[key][metric_name])
                                                  for metric_name in metric_names])

        display(Markdown(tabulate.tabulate(rows, headers, tablefmt=tablefmt)))


def display_multiple_model_results_multiple_types(group_name, results, tablefmt='github'):
    display(Markdown(f'## {group_name} Results'))
        
    key_groups = set([key[:-8] for key in next(iter(results.values())).keys()])
    metric_names = tuple(next(iter(next(iter(results.values())).values())).keys())
    headers = ['# Target Types', 'Model'] + [name + '&nbsp; ' * 12 for name in metric_names]
    
    for key_group in key_groups:
        display(Markdown(f'### {key_group}'))
        
        rows = []
        
        for n_types in range(1, 4):
            key = f'{key_group}-{n_types}-types'
            for model_name, model_results in results.items():
                rows.append([n_types, model_name] + [format_results(model_results[key][metric_name])
                                                      for metric_name in metric_names])
                
        display(Markdown(tabulate.tabulate(rows, headers, tablefmt=tablefmt)))

        
def display_multiple_model_results_multiple_types_replications(group_name, result_replications, tablefmt='github'):
    display(Markdown(f'## {group_name} Results'))
    
    ex_results = result_replications[0]
    key_groups = set([key[:-8] for key in next(iter(ex_results.values())).keys()])
    metric_names = tuple(next(iter(next(iter(ex_results.values())).values())).keys())
    headers = ['# Target Types', 'Model'] + [name + '&nbsp; ' * 12 for name in metric_names]
    
    for key_group in key_groups:
        display(Markdown(f'### {key_group}'))
        
        rows = []
        
        for n_types in range(1, 4):
            key = f'{key_group}-{n_types}-types'
            for model_name in ex_results:
                rows.append([n_types, model_name] + \
                            [format_replication_results([replication[model_name][key][metric_name] 
                                                         for replication in result_replications])
                                                      for metric_name in metric_names])
                
        display(Markdown(tabulate.tabulate(rows, headers, tablefmt=tablefmt)))
        

def format_result_or_result_list(task_result_or_list, print_std=True):
    if isinstance(task_result_or_list, list):
        return format_replication_results(task_result_or_list, print_std=print_std)
    
    return format_results(task_result_or_list, print_std=print_std)


def format_condition(condition, prev_condition):
    if condition == prev_condition:
        return ''
    
    return f'\\multirow{{2}}{{*}}{{\\textbf{{ {condition} }}}}'

def result_or_list_to_number_list(task_result_or_list, N=1024):
    if isinstance(task_result_or_list, list):
        if len(task_result_or_list) > 1:
            means = np.array([tr.mean for tr in task_result_or_list])
            return [means.mean(), means.std(), means.std() / (len(means) ** 0.5)]

        # len(task_result_or_list) == 0
        task_result_or_list = task_result_or_list[0]
    
    return [task_result_or_list.mean, task_result_or_list.std, task_result_or_list.std / (N ** 0.5)]

    
def multiple_results_to_df(all_results, df_headers=DF_HEADERS, N=1024):
    results_by_model_and_condition = defaultdict(list)
    
    for result_set in all_results:
        # handle the case where it's a list and I need to later average over it
        # flattens from a list of dicts of dicts to a dict of lists
        if isinstance(result_set, list):
            for result_set_replication in result_set:
                _parse_single_replication(results_by_model_and_condition, result_set_replication)

        # it's a dict
        else:
            _parse_single_replication(results_by_model_and_condition, result_set)
            
    # parse out results to a dataframe
    df_rows = []

    for model_and_condition_tuple, model_x_condition_results in results_by_model_and_condition.items():
        df_rows.append(list(model_and_condition_tuple) + result_or_list_to_number_list(model_x_condition_results, N=N))
        
    return pd.DataFrame(df_rows, columns=df_headers)

def _parse_single_replication(results_by_model_and_condition, result_set_replication):
    for model_name, model_results in result_set_replication.items():
        for condition_name, model_x_condition_results in model_results.items():
            results_by_model_and_condition[(model_name, condition_name)].append(model_x_condition_results['Accuracy'])
     