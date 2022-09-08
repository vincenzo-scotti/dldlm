from typing import Optional, List, Dict, Pattern, Tuple
import os
import math
import re
from tempfile import NamedTemporaryFile
from collections import Counter
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
