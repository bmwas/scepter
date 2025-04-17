# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import warnings
import os

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception as e:
    warnings.warn(f'Runing without matplotlib {e}')

color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
line_list = ['-', '--', '-.', ':']


def plot_multi_curves(x,
                      y,
                      show=False,
                      title=None,
                      save_path=None,
                      x_label=None,
                      y_label=None):
    '''
    Args:
        x: the x-axis data
        y: the y-axis data dict
            like: [{"data": np.ndarrays, "label": ""}]
        title: None
        show: False
        save_path: None
        x_label: None
        y_label: None
    Returns:
    '''
    if save_path is not None:
        plt.figure()

    x_max, x_min = np.max(x), np.min(x)

    max_num, min_num = 0, 0
    for y_id, data in enumerate(y):
        max_n = np.max(data['data'])
        min_n = np.min(data['data'])
        max_num = max_n if max_n > max_num else max_num
        min_num = min_n if min_n < min_num else min_num
        plt.plot(x,
                 data['data'],
                 linestyle=line_list[y_id % len(line_list)],
                 linewidth=2,
                 color=color_list[y_id % len(color_list)],
                 label=data['label'],
                 alpha=1.00)
        plt.title(title, loc='center')

    plt.legend(loc='upper right')
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)

    x_range = x_max - x_min
    if x_range == 0:
        x_step = 1
    else:
        x_step = x_range / 5
    if x_step > 0:
        plt.xticks(np.arange(x_min - x_step / 2, x_max + x_step / 2, x_step))
    y_step = (max_num - min_num) / 5 if max_num != min_num else 1
    if y_step > 0:
        plt.yticks(np.arange(min_num - y_step / 2, max_num + y_step / 2, y_step))
    plt.grid()
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.clf()
    plt.cla()
    plt.close()
    return True


def plt_curve(x,
              y,
              show=False,
              title=None,
              save_path=None,
              x_label=None,
              y_label=None):
    '''
    Args:
        x: the x-axis data
        y: the y-axis data dict
            like: [{"data": np.ndarrays, "label": ""}]
        title: None
        show: False
        save_path: None
        x_label: None
        y_label: None
    Returns:
    '''
    return plot_multi_curves(x, [{
        'data': y,
        'label': 'y'
    }],
                             show=show,
                             title=title,
                             save_path=save_path,
                             x_label=x_label,
                             y_label=y_label)


def plot_results(plot_data, save_folder):
    # Only plot the 'all' curve for clarity
    if 'all' in plot_data:
        label = 'all'
        curve_data = [[step, value] for step, value in plot_data['all'].items()]
        curve_data.sort(key=lambda x: x[0])
        steps = [step for step, value in curve_data]
        value = [value for step, value in curve_data]
        k_y = [{'data': np.array(value), 'label': label}]
        save_path = os.path.join(save_folder, f"{label}.png")
        with open(save_path, 'w') as local_file:
            plot_multi_curves(x=np.array(steps),
                              y=k_y,
                              x_label='steps',
                              y_label='validation loss',
                              title=f"{label} validation loss",
                              save_path=local_file)
    # Optionally, for meta curves, plot only if there is more than one unique value
    # (not implemented here for clarity)
    if len(steps) > 0:
        return True
    return False
