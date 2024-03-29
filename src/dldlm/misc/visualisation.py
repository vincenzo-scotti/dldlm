from typing import Optional, List, Dict, Pattern, Tuple
import os
import math
import re
from tempfile import NamedTemporaryFile
from collections import Counter

import numpy as np

import torch
from matplotlib import pyplot as plt
from matplotlib import colors as cols
import torchvision
import plotly.graph_objects as go
from torch.utils.tensorboard import SummaryWriter


def log_word_stats(
        data: Dict[str, Tuple[Counter, Counter]],
        top_n: int = 15,
        group_id_regex: Pattern[str] = re.compile(r'<[|]latentcode(\d+)[|]>'),
        tb_writer: Optional[SummaryWriter] = None,
        sub_tag: Optional[str] = None,
        step: Optional[int] = None,
        dest_dir: Optional[str] = None,
        file_name: Optional[str] = None
):
    text = dict()
    for i, (stat, stat_tag) in enumerate((('with stop-word', 'w_sw'), ('without stop words', 'wo_sw'))):
        tmp_text = f"{sub_tag} - {stat}\n" if sub_tag is not None else f"{stat}\n"
        for key in data:
            latent = int(group_id_regex.search(key).group(1)) if group_id_regex.match(key) else None
            tmp_text += (f"\tLatent: {latent}\n\t\t" if sub_tag is not None else f"Latent: {latent}\n\t") + \
                    ("\n\t\t" if sub_tag is not None else "\n\t").join(
                        f"{item}: {count}" for item, count in data[key][i].most_common(top_n)
                    ) + "\n"

        text[stat_tag] = tmp_text

        if tb_writer is not None:
            tag = f'Action-word stats/{sub_tag} - {stat}' if sub_tag is not None else f'Action-word stats/{stat}'
            tb_writer.add_text(tag,  f"<pre>{tmp_text}</pre>", step)
        if dest_dir is not None and file_name is not None:
            if not os.path.exists(dest_dir):
                os.mkdir(dest_dir)
            file_path = os.path.join(dest_dir, file_name)
            with open(file_path, 'a') as f:
                print(tmp_text, file=f)

    return text


def plot_word_stats(
        data: Dict[str, Tuple[Counter, Counter]],
        top_n: int = 15,
        group_id_regex: Pattern[str] = re.compile(r'<[|]latentcode(\d+)[|]>'),
        tb_writer: Optional[SummaryWriter] = None,
        sub_tag: Optional[str] = None,
        step: Optional[int] = None,
        dest_dir: Optional[str] = None,
        file_name: Optional[str] = None
):
    # TODO fix issue with empty counter and additional axes that are not used
    figure = dict()
    for i, (stat, stat_tag) in enumerate((('TF', 'tf'), ('TF-IDF', 'tf_idf'))):
        n_items = len(data)
        n_cols = 4
        n_rows = int(math.ceil(n_items / n_cols))
        tmp_figure, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(24, 1 + (n_rows * 7)), sharex=True)
        axes = axes.flatten()
        for key_idx, key in enumerate(data):
            try:
                items, counts = list(zip(*data[key][i].most_common(top_n)))
            except ValueError:
                items = tuple()
                counts = tuple()
            ax = axes[key_idx]
            ax.barh(items, counts, height=0.7)
            latent = int(group_id_regex.search(key).group(1)) if group_id_regex.match(key) else None
            ax.set_title(f"Latent: {latent}", fontdict={"fontsize": 20})
            ax.invert_yaxis()
            ax.tick_params(axis="both", which="major", labelsize=20)
            for pos in "top right left".split():
                ax.spines[pos].set_visible(False)
            if sub_tag is not None:
                tmp_figure.suptitle(f"Latent actions word stats ({sub_tag})", fontsize=32)
        plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)

        figure[stat_tag] = tmp_figure

        if tb_writer is not None:
            if sub_tag is not None:
                tag = f'Latent actions word stats ({sub_tag} - {stat})'
            else:
                tag = f'Latent actions word stats ({stat})'
            tb_writer.add_figure(tag, tmp_figure, step)
        if dest_dir is not None and file_name is not None:
            if not os.path.exists(dest_dir):
                os.mkdir(dest_dir)
            file_path = os.path.join(dest_dir, f'{stat_tag}_{file_name}')
            tmp_figure.savefig(file_path)

    return figure


def log_traces(
        data: List[List[str]],
        group_id_regex: Pattern[str] = re.compile(r'<[|]latentcode(\d+)[|]>'),
        tb_writer: Optional[SummaryWriter] = None,
        sub_tag: Optional[str] = None,
        step: Optional[int] = None,
        dest_dir: Optional[str] = None,
        file_name: Optional[str] = None
):
    text = "\n".join(
        ", ".join(
            f'{group_id_regex.search(elem).group(1) if group_id_regex.match(elem) else None}'
            for elem in trace)
        for trace in data)

    if tb_writer is not None:
        tag = f'Latent action traces/{sub_tag}' if sub_tag is not None else 'Latent action traces'
        tb_writer.add_text(tag,  f"<pre>{text}</pre>", step)
    if dest_dir is not None and file_name is not None:
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)
        file_path = os.path.join(dest_dir, file_name)
        with open(file_path, 'w') as f:
            print(text, file=f)

    return text


def plot_traces(
        data: List[List[str]],
        group_id_regex: Pattern[str] = re.compile(r'<[|]latentcode(\d+)[|]>'),
        tb_writer: Optional[SummaryWriter] = None,
        sub_tag: Optional[str] = None,
        step: Optional[int] = None,
        dest_dir: Optional[str] = None,
        file_name: Optional[str] = None
):
    node_traces = [list(enumerate(trace)) for trace in data]
    pair_counts = Counter(pair for trace in node_traces for pair in zip(trace[:-1], trace[1:]))
    node_decoder = list(set(node for pair in pair_counts for node in pair))
    node_encoder = {node: i for i, node in enumerate(node_decoder)}

    source_nodes, target_nodes = list(zip(*pair_counts))
    values = [pair_counts[(source, target)] for source, target in zip(source_nodes, target_nodes)]
    source_nodes = [node_encoder[node] for node in source_nodes]
    target_nodes = [node_encoder[node] for node in target_nodes]

    labels = [str(group_id_regex.search(lbl).group(1) if group_id_regex.match(lbl) else None) for _, lbl in node_decoder]
    n_labels = len(set(labels))
    cmap = plt.get_cmap('Spectral')
    colors_dict = {lbl: cols.rgb2hex(cmap(i / n_labels)) for i, lbl in enumerate(set(labels))}
    colors = [colors_dict[lbl] for lbl in labels]

    figure = go.Figure(
        data=[go.Sankey(
            node=dict(pad=10, thickness=20, line=dict(color="black", width=0.5), label=labels, color=colors),
            link=dict(source=source_nodes, target=target_nodes, value=values)
        )]
    )
    figure.update_layout(font_size=10)

    if tb_writer is not None:
        tag = f'Latent action traces ({sub_tag})' if sub_tag is not None else 'Latent action traces'
        with NamedTemporaryFile('wb', suffix='.png') as f:
            figure.write_image(f.name)
            f.seek(0)
            img_tensor = torchvision.io.read_image(f.name)
        tb_writer.add_image(tag, img_tensor, step)
    if dest_dir is not None and file_name is not None:
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)
        file_path = os.path.join(dest_dir, file_name)
        figure.write_image(file_path)

    return figure


def log_generated_response(
        data: List[Dict],
        group_id_regex: Pattern[str] = re.compile(r'<[|]latentcode(\d+)[|]>'),
        tb_writer: Optional[SummaryWriter] = None,
        sub_tag: Optional[str] = None,
        step: Optional[int] = None,
        dest_dir: Optional[str] = None,
        file_name: Optional[str] = None
):
    def str_sample(**kwargs):
        return f"Corpus:\n\t{kwargs['corpus']}, Split: {kwargs['split']}, Dialogue idx: {kwargs['conversation_idx']}\n\n" + \
               f"Prompted context:\n\t{repr(kwargs['context'])}\n\n" + \
               "Generated responses:\n\t" + "\n\t".join(f"Latent {group_id_regex.search(latent).group(1) if group_id_regex.match(latent) else None}: {kwargs['generated_responses'][latent]}" for latent in kwargs['generated_responses']) + "\n\n" + \
               f"Original response:\n\tLatent {group_id_regex.search(kwargs['latent']).group(1)  if group_id_regex.match(kwargs['latent']) else None}: {kwargs['response']}"

    text = "\n\n".join(str_sample(**sample) for sample in data)

    if tb_writer is not None:
        tag = f'Generated responses/{sub_tag}' if sub_tag is not None else 'Generated responses'
        tb_writer.add_text(tag,  f"<pre>{text}</pre>", step)
    if dest_dir is not None and file_name is not None:
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)
        file_path = os.path.join(dest_dir, file_name)
        with open(file_path, 'w') as f:
            print(text, file=f)

    return text


def log_latents_count(
        data: Dict[str, int],
        group_id_regex: Pattern[str] = re.compile(r'<[|]latentcode(\d+)[|]>'),
        tb_writer: Optional[SummaryWriter] = None,
        sub_tag: Optional[str] = None,
        step: Optional[int] = None,
        dest_dir: Optional[str] = None,
        file_name: Optional[str] = None
):
    text = (f"{sub_tag}\n\t" if sub_tag is not None else str()) + \
           ("\n\t" if sub_tag is not None else "\n").join(
               f"Latent {group_id_regex.search(latent).group(1) if group_id_regex.match(latent) else None}: {count}"
               for latent, count in data.items()
           )

    if tb_writer is not None:
        tag = f'Latent code occurrences/{sub_tag}' if sub_tag is not None else 'Latent code occurrences'
        tb_writer.add_text(tag,  f"<pre>{text}</pre>", step)
    if dest_dir is not None and file_name is not None:
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)
        file_path = os.path.join(dest_dir, file_name)
        with open(file_path, 'a') as f:
            print(text, file=f)

    return text


def log_correlations(
        data: Dict[str, Dict[str, Dict[str, torch.Tensor]]],
        tb_writer: Optional[SummaryWriter] = None,
        sub_tag: Optional[str] = None,
        step: Optional[int] = None,
        dest_dir: Optional[str] = None,
        file_name: Optional[str] = None
):
    text = f"{sub_tag}\n\n" if sub_tag is not None else str()
    for tgt, correlations in data.items():
        text += f"\t{tgt}:\n"
        for subset, correlations in correlations.items():
            text += f"\t\t{subset}:\n"
            for tgt_value, (support, std, avg) in correlations.items():
                text += f"\t\t\t{tgt_value} ({support}): " + \
                        ", ".join(f"{avg:.4f} ({std:.4f})" for avg, std in zip(avg.tolist(), std.tolist())) + \
                        "\n"
        text += '\n'

    if tb_writer is not None:
        tag = f'Observed correlations/{sub_tag}' if sub_tag is not None else 'Observed correlations'
        tb_writer.add_text(tag,  f"<pre>{text}</pre>", step)
    if dest_dir is not None and file_name is not None:
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)
        file_path = os.path.join(dest_dir, file_name)
        with open(file_path, 'w') as f:
            print(text, file=f)

    return text


def plot_correlations(
        data: Dict,
        tb_writer: Optional[SummaryWriter] = None,
        sub_tag: Optional[str] = None,
        step: Optional[int] = None,
        dest_dir: Optional[str] = None,
        file_name: Optional[str] = None
):
    figure = dict()
    # Gather considered subsets
    sub_sets = set(sub_set for correlations in data.values() for sub_set in correlations)
    # Correlations order  # TODO find better solution
    correlations = ['sentiment', 'dialogue_act', 'speaker']
    # Generate one plot per sub-set
    for i, sub_set in sub_sets:
        # Get tmp tag
        tmp_tag = sub_set.lower().replace(' ', '_')
        # Get list of supported correlations
        tmp_correlations = [key for key in correlations if sub_set in data[key]]
        # Get current subset data
        tmp_data = {correlation: data[correlation][sub_set] for correlation in tmp_correlations}

        n_styles = tmp_data[tmp_correlations[0]][list(tmp_data[tmp_correlations[0]])[0]][-1].shape(0)  # TODO find better solution

        mat = {
            key: torch.vstack([
                torch.tensor([avg[i] * count for count, std, avg in values.values()]) /
                torch.tensor([avg[i] * count for count, std, avg in values.values()]).sum()
                for i in range(n_styles)
            ])
            for key, values in tmp_data.items()
        }

        counts = {
            key: torch.tensor([count for count, std, avg in values.values()]).reshape(1, -1)
            for key, values in tmp_data.items()
        }

        p_dist = (
            torch.tensor([count * avg for count, std, avg in tmp_data[tmp_correlations[0]].values()]) /
            torch.tensor([count for count, std, avg in tmp_data[tmp_correlations[0]].values()]).sum()
        )

        tot = sum(count for count, std, avg in tmp_data[tmp_correlations[0]].values())

        #
        width = sum(1 + m.shape[-1] for m in mat.values())

        n_cols = 1 + len(tmp_correlations)
        n_rows = 2
        tmp_figure, axes = plt.subplots(
            nrows=n_rows, ncols=n_cols, figsize=(width * 0.8, 12 * 0.8), sharex='col', sharey='row',
            gridspec_kw={'wspace': .5, 'hspace': .25, 'width_ratios': [len(data[correlation][sub_set]) for correlation in tmp_correlations] + [1],
                         'height_ratios': [n_styles, 1]}
        )
        axes = axes.flatten()
        for key_idx, key in enumerate(tmp_correlations + ['Labels']):
            axt = axes[key_idx]
            axb = axes[key_idx + n_cols]
            if key_idx < n_cols:
                if key_idx < n_cols - 1:
                    p_dist_img = axt.matshow(mat[key], cmap=plt.cm.Blues, vmin=0, vmax=1)
                    axt.set_xticklabels([])
                    axt.set_xticks(range(len(data[key][sub_set])))
                    axt.set_yticks(range(n_styles))
                    if key_idx == 0:
                        axt.set_yticklabels([f'Latent {i}' for i in range(n_styles)])
                        axt.set_ylabel('Latent-probability distribution in group')
                    else:
                        axt.yaxis.set_ticks_position('none')
                        # axt.sharey(axes[0])
                    # axt.set_title(f"{key}", fontdict={"fontsize": 20})
                    axt.set_xlabel(f'{key}')
                    axt.xaxis.set_ticks_position('none')
                    cax = tmp_figure.add_axes([axt.get_position().x0,
                                               axt.get_position().y0 - 0.04,
                                               axt.get_position().width,
                                               0.015])

                    # Plot horizontal colorbar on created axes
                    plt.colorbar(p_dist_img, orientation="horizontal", cax=cax, ticks=[0.0, 1.0])

                    axb.matshow(counts[key], cmap=plt.cm.Reds, vmin=0, vmax=tot)
                    axb.set_xticklabels([*data[key][sub_set]], rotation=45, ha='right')
                    axb.set_xticks(range(len(data[key][sub_set])))
                    axb.xaxis.set_ticks_position('bottom')
                    axb.yaxis.set_ticks_position('none')
                    axb.set_yticklabels([])
                    # axb.set_title(f"Support", fontdict={"fontsize": 20})
                    axb.set_ylabel('Support', rotation=0, ha='right', va='center')
                    axb.set_xlabel(f'{key}')
                    for (i, j), z in np.ndenumerate(counts[key].numpy()):
                        axb.text(j, i, f'{z}', ha='center', va='center', color='white' if z >= tot / 2 else 'black')
                else:
                    p_dist_img = axt.matshow(p_dist, cmap=plt.cm.Greens, vmin=0, vmax=1)
                    axt.set_xticklabels([])
                    axt.set_xticks(range(1))
                    axt.set_yticks(range(n_styles))
                    axt.xaxis.set_ticks_position('none')
                    axt.yaxis.set_ticks_position('none')
                    # axt.set_title(f"Posterior distribution", fontdict={"fontsize": 20})
                    axt.set_ylabel(f"Posterior latent probability distribution")
                    cax = tmp_figure.add_axes([axt.get_position().x1 + 0.01,
                                               axt.get_position().y0,
                                               0.0075,
                                               axt.get_position().height])
                    plt.colorbar(p_dist_img, cax=cax, ticks=[0.0, 1.0])  # ax=axt)
                    axt.sharey(axes[0])

                    count_img = axb.matshow([[tot]], cmap=plt.cm.Reds, vmin=0, vmax=tot)
                    axb.xaxis.set_ticks_position('none')
                    axb.yaxis.set_ticks_position('none')
                    axb.set_xticklabels([])
                    axb.set_yticklabels([])
                    # axb.set_title(f"Support", fontdict={"fontsize": 20})
                    axb.text(0, 0, f'{tot}', ha='center', va='center', color='white')
                    # axb.sharex(axt)

        tmp_figure.suptitle(f"Correlation", fontsize=32)
        plt.tight_layout()
        #

        figure[tmp_tag] = tmp_figure

        if tb_writer is not None:
            if sub_tag is not None:
                tag = f'Observed correlations ({sub_tag} - {tmp_tag})'
            else:
                tag = f'Observed correlations ({tmp_tag})'
            tb_writer.add_figure(tag, tmp_figure, step)
        if dest_dir is not None and file_name is not None:
            if not os.path.exists(dest_dir):
                os.mkdir(dest_dir)
            file_path = os.path.join(dest_dir, f'{tmp_tag}_{file_name}')
            tmp_figure.savefig(file_path)

    return figure


    # figure = dict()
    # # Gather considered subsets
    # sub_sets = set(sub_set for correlations in data.values() for sub_set in correlations)
    # # Correlations order  # TODO find better solution
    # correlations = ['sentiment', 'dialogue_act', 'speaker']
    # # Generate one plot per sub-set
    # for i, sub_set in sub_sets:
    #     # Get tmp tag
    #     tmp_tag = sub_set.lower().replace(' ', '_')
    #     # Get list of supported correlations
    #     tmp_correlations = [key for key in correlations if sub_set in data[key]]
    #
    #     width = sum(1 + len(data[key][sub_set]) for key in tmp_correlations)
    #
    #     n_cols = 1 + len(tmp_correlations)
    #     n_rows = 2
    #     tmp_figure, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(width, 12), sharey=True)
    #     # axes = axes.flatten()
    #
    #     # Work on computed correlations
    #     for correlation in tmp_correlations:
    #         x_labels = [label for ]
    #
    #
    #     for key_idx, key in enumerate(data):
    #         try:
    #             items, counts = list(zip(*data[key][i].most_common(top_n)))
    #         except ValueError:
    #             items = tuple()
    #             counts = tuple()
    #         ax = axes[key_idx]
    #         ax.barh(items, counts, height=0.7)
    #         latent = int(group_id_regex.search(key).group(1)) if group_id_regex.match(key) else None
    #         ax.set_title(f"Latent: {latent}", fontdict={"fontsize": 20})
    #         ax.invert_yaxis()
    #         ax.tick_params(axis="both", which="major", labelsize=20)
    #         for pos in "top right left".split():
    #             ax.spines[pos].set_visible(False)
    #         if sub_tag is not None:
    #             tmp_figure.suptitle(f"Latent actions word stats ({sub_tag})", fontsize=32)
    #     plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    #
    #     figure[tmp_tag] = tmp_figure
    #
    #     if tb_writer is not None:
    #         if sub_tag is not None:
    #             tag = f'Observed correlations ({sub_tag} - {tmp_tag})'
    #         else:
    #             tag = f'Observed correlations ({tmp_tag})'
    #         tb_writer.add_figure(tag, tmp_figure, step)
    #     if dest_dir is not None and file_name is not None:
    #         if not os.path.exists(dest_dir):
    #             os.mkdir(dest_dir)
    #         file_path = os.path.join(dest_dir, f'{tmp_tag}_{file_name}')
    #         tmp_figure.savefig(file_path)
    #
    # return figure
