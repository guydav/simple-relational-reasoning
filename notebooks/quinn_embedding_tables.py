import itertools
import numpy as np
import pandas as pd
import tabulate

from collections import defaultdict
from IPython.display import display, Markdown


MODEL_ORDERS = [f'{model}-{dataset}' for (dataset, model) 
                in itertools.product(('saycam(S)', 'imagenet', 'random'), ('mobilenet', 'resnext'))]

HEADERS = ['Stimulus Rendering', 'Target Type'] + MODEL_ORDERS
DF_HEADERS = ['relation', 'rendering', 'n_target_types', 'model_name', 'dataset', 'acc_mean', 'acc_std', 'acc_sem'] 

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


def result_or_list_to_numbers(task_result_or_list, N=1024):
    if isinstance(task_result_or_list, list):
        means = np.array([tr.mean for tr in task_result_or_list])
        return [means.mean(), means.std(), means.std() / (len(means) ** 0.5)]
    
    return [task_result_or_list.mean, task_result_or_list.std, task_result_or_list.std / (N ** 0.5)]


def format_condition(condition, prev_condition):
    if condition == prev_condition:
        return ''
    
    return f'\\multirow{{2}}{{*}}{{\\textbf{{ {condition} }}}}'

    
def table_per_relation_multiple_results(all_results, tablefmt='github', model_orders=MODEL_ORDERS, 
                                        n_types_to_print=(1, 2), headers=HEADERS, df_headers=DF_HEADERS, 
                                        print_std=True, N=1024):
    models_datasets = list(all_results[0].keys())
    
    results_by_relation = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    for result_set in all_results:
        # TODO: handle the case where it's a list and I need to average over it
        if isinstance(result_set, list):
            for result_set_replication in result_set:
                for model_and_dataset, model_and_dataset_results in result_set_replication.items():
                    for full_condition_name, condition_results in model_and_dataset_results.items():
                        relation, condition, n_types = parse_condition_name(full_condition_name)
                        if model_and_dataset not in results_by_relation[relation][condition][n_types]:
                            results_by_relation[relation][condition][n_types][model_and_dataset] = []
                        results_by_relation[relation][condition][n_types][model_and_dataset].append(condition_results['Accuracy'])
            
        # it's a dict
        else:
            for model_and_dataset, model_and_dataset_results in result_set.items():
                for full_condition_name, condition_results in model_and_dataset_results.items():
                    relation, condition, n_types = parse_condition_name(full_condition_name)
                    results_by_relation[relation][condition][n_types][model_and_dataset] = condition_results['Accuracy']
                
    # parse out results by relation to a set of tables
    all_df_rows = []
    
    for relation, relation_results in results_by_relation.items():
        display(Markdown(f'## {relation}'))
        rows = []
        prev_condition = None
        for condition, condition_results in relation_results.items():
            for n_types in n_types_to_print:
                condition_and_n_types_results = condition_results[n_types]
                types_name = n_types == 1 and 'Same Target' or 'Different Target'
                formatted_condition = format_condition(condition, prev_condition)
                prev_condition = condition
                formatted_results = [format_result_or_result_list(condition_and_n_types_results[model_and_dataset], 
                                                                                        print_std=print_std) 
                                                           for model_and_dataset in model_orders]
                row = [formatted_condition, types_name] + formatted_results
                rows.append(row)
                
                df_row_prefix = [simplify(relation), simplify(condition), n_types]
                
                df_rows = [df_row_prefix + [s.replace('(S)', '')  for s in (model_and_dataset.split('-'))] + 
                           result_or_list_to_numbers(condition_and_n_types_results[model_and_dataset], N=N)
                           for model_and_dataset in model_orders]
                all_df_rows.extend(df_rows)
            
        display(Markdown(tabulate.tabulate(rows, headers, tablefmt=tablefmt)))
        
    return pd.DataFrame(all_df_rows, columns=df_headers)
    