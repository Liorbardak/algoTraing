
import warnings

import matplotlib

# MATPLOTLIB_BACKEND = matplotlib.get_backend()
# matplotlib.use('Agg')



import re
import matplotlib.pyplot as plt

# from plotly.basedatatypes import BaseFigure
# import plotly.graph_objects as go
import base64
from io import BytesIO
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import pprint
from typing import Set
import types
import os
#import pwd
from html import escape as html_escape_orig

from typing import Any, List, Dict, Union, Optional
import psutil


def html_escape(s: Any) -> str:
    return html_escape_orig(str(s), quote=False)


def do_not_use_cpu0():
    do_not_use_cpus({0})


def do_not_use_cpus(cpus: Set):
    p = psutil.Process()
    all_but_first = [x for x in p.cpu_affinity() if x not in cpus]
    p.cpu_affinity(all_but_first)
    print(f"do_not_use_cpu0 using: {p.cpu_affinity()}")


def get_username():
    try:
        return pwd.getpwuid(os.getuid())[0]
    except:
        return "unknown"


class HtmlReport:
    def __init__(self, add_defaults=True):
        self.sections = []
        if add_defaults:
            self.add_header()
            self.sections.append(f"<body><main>")



    def add_header(self):
        try:
            with open(
                Path(__file__).parent.parent / "utils/web/static/styles/style.css", "r"
            ) as f:
                styles = "".join(f.read().splitlines())
        except FileNotFoundError:
            styles = ""
        header = f"""
        <head>
            <!-- Required meta tags -->
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
            <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">
            <link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.6.3/css/all.css" integrity="sha384-UHRtZLI+pbxtHCWp1t77Bi1L4ZtiqrqD80Kn4Z8NTSRyMA2Fd33n5dQ8lWUE00s/" crossorigin="anonymous">
            <link rel="stylesheet" href="https://unpkg.com/bootstrap-table@1.18.3/dist/bootstrap-table.min.css">
            <link rel="stylesheet" type="text/css" href="/data/transfer/planning/prediction/benchmark/bootstrap/extensions/filter-control/bootstrap-table-filter-control.css">
            <style>{styles}</style>
        </head>
        """
        self.sections.append(f"{header}")

    def add_js_scripts(self):
        js_scripts = """
            <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.5.1/jquery.min.js" integrity="sha512-bLT0Qm9VnAYZDflyKcBaQ2gg0hSYNQrJ8RilYldYQ1FxQYoCLtUjuuRuZo+fjqhx/qtq/1itJ0C2ejDxltZVFg==" crossorigin="anonymous"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.7/umd/popper.min.js" integrity="sha384-UO2eT0CpHqdSJQ6hJty5KVphtPhzWj9WO1clHTMGa3JDZwrnQq4sF86dIHNDz0W1" crossorigin="anonymous"></script>
            <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.min.js" integrity="sha384-JjSmVgyd0p3pXB1rRibZUAYoIIy6OrQ6VrjIEaFf/nJGzIxFDsf4x0xIM+B07jRM" crossorigin="anonymous"></script>
            <script src="https://unpkg.com/bootstrap-table@1.18.3/dist/bootstrap-table.min.js"></script>
            <script src="/data/transfer/planning/prediction/benchmark/bootstrap/extensions/filter-control/bootstrap-table-filter-control.js"></script>
            """
        self.sections.append(f"{js_scripts}")

    def add_html(self, html: str):
        self.sections.append(f"{html}")

    def add_line(self, line: str):
        self.sections.append(f"{html_escape(line)}<br></br>")

    def add_title(self, title: str, level: int = 2):
        self.sections.append(
            f"<div><br></br><h{level}>{html_escape(title)}</h{level}></div>"
        )

    def add_link(
        self,
        title: str,
        link: str = None,
        text_align: str = "left",
        margin=0,
        br: bool = True,
        div: bool = True,
    ):
        link = link or title
        html = f'<a href="{link}">{html_escape(title)}</a>'
        if div:
            html = f'<div style="text-align:{text_align};margin-{text_align}:{margin}%"> {html} </div>'
        if br:
            html += "<br></br>"
        self.sections.append(html)

    def add_df(
        self,
        title: str,
        df: pd.DataFrame,
        float_format=None,
        render_links=False,
        escape=True,
        searchable=False,
        searchable_columns=None,
    ):
        tbl = df.to_html(
            float_format=float_format, render_links=render_links, escape=escape
        )
        if searchable:
            searchable_columns = searchable_columns or []
            searchable_tbl_tag = """
            <table
                border="1"
                class="table table-striped table-bordered table-hover"
                data-search="true" data-toggle="table"  
                data-pagination="true"  
                data-show-columns="true"
                data-filter-control="true"  
                data-show-toggle="true"
                data-show-columns-toggle-all="true"
                data-show-pagination-switch="true"
                data-sortable="true"
            >"""

            thead_tag = f"""
            <thead>
                <tr>
                <th data-sortable="true"></th>
            """
            for col_name in df.columns:
                if col_name in searchable_columns:
                    thead_tag += f"""<th data-field="{col_name}" data-filter-control="select" data-filter-strict-search="true" data-sortable="true">{col_name}</th>"""
                else:
                    thead_tag += f"""<th data-sortable="true">{col_name}</th>"""
            thead_tag += """
                </tr>
            </thead>
            """

            tbl = re.sub(r"(?s)<table.*?>", searchable_tbl_tag, tbl)
            tbl = re.sub(r"(?s)<thead.*?>.*?</thead>", thead_tag, tbl)

        self.add_title(title=title)
        self.sections.append(f"<div>" f"{tbl}" f"<br></br></div>")

    def add_figure(self, title: str, fig: Any):
        try:
            self.add_matplot_figure(title=title, fig=fig)
        except:
            try:
                self.add_plotly_figure(title=title, fig=fig)
            except:
                raise

    def add_png_encoded_as_hexstring(self, title: Optional[str], png_as_bytes):
        encoded = base64.b64encode(png_as_bytes).decode("utf-8")
        html = f"<div><img src='data:image/png;base64,{encoded}'></div><br></br>"
        if title is not None:
            html = f"<div><h{2}>{html_escape(title)}</h{2}>{html}</div>"
        self.sections.append(html)

    def add_matplot_figure(
        self, title: Optional[str], fig: Any, dpi: Union[str, float] = "figure"
    ):  # fig - matplot figure
        tmpfile = BytesIO()
        fig.savefig(tmpfile, format="png", dpi=dpi)
        png_as_bytes = tmpfile.getvalue()
        self.add_png_encoded_as_hexstring(title=title, png_as_bytes=png_as_bytes)

    # def add_plotly_figure(self, title: Optional[str], fig: BaseFigure):
    #     html = fig.to_html(full_html=False) + "<br></br>"
    #     if title is not None:
    #         html = f"<div><h{2}>{html_escape(title)}</h{2}>{html}</div>"
    #     self.sections.append(html)
    #
    # def add_plotly_figure_as_png(self, title: Optional[str], fig: BaseFigure):
    #     # this require to install kaleido
    #     # pip install -U kaleido
    #
    #     try:
    #         import kaleido
    #     except:
    #         print(
    #             "!!!!!!!!!!! WARNING add_plotly_figure_as_png *converted to add_plotly_figure* "
    #             'because "kaleido" is not install. try:'
    #         )
    #         print("!!!!!!!!!!! pip install -U kaleido")
    #         return self.add_plotly_figure(title=title, fig=fig)
    #
    #     self.add_png_encoded_as_hexstring(
    #         title=title,
    #         png_as_bytes=fig.to_image(
    #             format="png", width=1600, height=800, validate=True
    #         ),
    #     )

    def add_dict(self, d: Dict):
        for k, v in d.items():
            self.add_line(f"{html_escape(k)}: {html_escape(v)}")

    def to_html_string(self):
        return "".join(self.sections)

    def to_file_obj(self, f):
        f.write("<!doctype html><html>")
        f.write(self.to_html_string())
        f.write("</main></body></html>")

    def to_file(self, out_path: str):
        self.add_js_scripts()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            self.to_file_obj(f)


    def add_anchor(self, anchor: str, text: str = "", div: bool = False):
        html = f'<a name="{anchor}">{html_escape(text)}</a>'
        if div:
            html = f"<div>{html}</div>"
        self.sections.append(html)

    def show(self):
        import tempfile
        import webbrowser
        import os

        f, fpath = tempfile.mkstemp(suffix=".html", text=True)
        os.close(f)
        self.to_file(fpath)
        new = 2  # open in a new tab, if possible
        url = f"file://{fpath}"
        webbrowser.open(url, new=new)


class TrainLog:
    def __init__(self, name: str, loss_functions_cnt: int):
        self.name = name
        self.validation_losses = [[] for _ in range(loss_functions_cnt)]
        self.train_losses = [[] for _ in range(loss_functions_cnt)]
        self.info = []
        self.aux_plot_data: List[Dict[str, float]] = []

    def add_aux_values(self, values: np.array):
        if values:
            self.aux_plot_data.append(values)

    def add_str(self, s: str):
        self.info.append(s)

    def add_validation_loss(
        self, loss_func_idx: int, epoch: float, loss: float, learning_rate: float
    ):
        self.validation_losses[loss_func_idx].append((epoch, loss, learning_rate))

    def add_train_loss(self, loss_func_idx: int, epoch: float, loss: float):
        self.train_losses[loss_func_idx].append((epoch, loss))

    def aux_figure(self, format_plotly=True):
        title = f"auxiliary plot"
        df = pd.DataFrame(data=self.aux_plot_data)

        if format_plotly:
            # this is the elegant way but it fails sometimes
            # import plotly.express as px
            # return px.scatter(data_frame=df, title=title)
            x = df.index.values
            fig = go.Figure()
            for c in df.columns:
                fig.add_trace(go.Scatter(x=x, y=df[c].values, mode="lines", name=c))
            fig.update_layout(title=title, xaxis_title="epoch", yaxis_title="aux")
            return fig
        else:
            fig = df.plot()
            plt.title(title)
            return fig.get_figure()

    def loss_figure(self, loss_func_idx: int, format_plotly=True):

        vx, vl, vtr = zip(*self.validation_losses[loss_func_idx])
        tx, tl = zip(*self.train_losses[loss_func_idx])

        trx = []
        tryy = []
        for i, lr in enumerate(vtr[1:]):
            if lr != vtr[i]:
                trx.append(vx[i + 1])
                tryy.append(vl[i + 1])

        if format_plotly:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=vx, y=vl, mode="lines", name="validation_loss"))
            fig.add_trace(go.Scatter(x=tx, y=tl, mode="lines", name="training_loss"))
            fig.add_trace(
                go.Scatter(x=trx, y=tryy, mode="markers", name="learning_rate_changed")
            )
            fig.update_layout(
                title=f"train/validation losses - loss func: {loss_func_idx}: {self.name}",
                xaxis_title="epoch",
                yaxis_title="loss",
            )
        else:
            fig = plt.figure()
            plt.plot(vx, vl, ".b", tx, tl, ".r", trx, tryy, "*y")
            plt.ylabel("loss")
            plt.xlabel("epoch")
            plt.title(
                f"train/validation losses - loss func: {loss_func_idx}: {self.name}"
            )

        return fig

    def loss_funcs_cnt(self):
        return len(self.train_losses)

    def add_one_to_report(self, loss_func_idx: int, report: HtmlReport):
        report.add_figure(
            title=f"{self.name} (loss function: {loss_func_idx})",
            fig=self.loss_figure(loss_func_idx),
        )

    def add_to_report(self, report: HtmlReport):
        for i in range(len(self.train_losses)):
            self.add_one_to_report(loss_func_idx=i, report=report)

    def add_aux_plot_to_reprt(self, report: HtmlReport):
        the_aux_fig = self.aux_figure()
        if the_aux_fig:
            report.add_figure(title="aux plot", fig=the_aux_fig)


def plot_tensorboard(dir, tag, global_step, val):
    import os
    import tensorflow as tf

    if not os.path.exists(dir):
        os.makedirs(dir)

    if tf.__version__.startswith("2"):
        writer = tf.summary.create_file_writer(dir)
        with writer.as_default():
            tf.summary.scalar(tag, val, step=global_step)
            writer.flush()
    else:
        warnings.warn("tf.summary.FileWriter is deprecated as of tensorflow 2.")
        writer = tf.summary.FileWriter(dir)
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])
        writer.add_summary(summary, global_step)


def example_train_log():
    ti = TrainLog("test", 1)
    ti.add_train_loss(0, 0.33, 12)
    ti.add_train_loss(0, 0.66, 7)
    ti.add_train_loss(0, 1, 3)
    ti.add_train_loss(0, 1.33, 1)
    ti.add_train_loss(0, 1.66, 0.9)
    ti.add_train_loss(0, 2, 1.1)
    ti.add_train_loss(0, 2.33, 0.8)
    ti.add_train_loss(0, 2.66, 0.7)
    ti.add_train_loss(0, 3, 1)
    ti.add_train_loss(0, 3.33, 0.6)
    ti.add_train_loss(0, 3.66, 0.8)
    ti.add_train_loss(0, 4, 1.2)
    ti.add_validation_loss(0, 1, 6, 1e-3)
    ti.add_validation_loss(0, 2, 1.5, 1e-3)
    ti.add_validation_loss(0, 3, 1.2, 1e-4)
    ti.add_validation_loss(0, 4, 1.7, 1e-4)

    fig = ti.loss_figure(0)
    fig.show()


def classification_report_to_df(cr: str) -> pd.DataFrame:
    """
    :param cr: a classifcation report outputed by sklearn.metrics.classification_report
    :return: data frame representing the report
    """

    classes = []
    precision = []
    recall = []
    f1_score = []
    support = []

    cr = cr.replace("macro avg", "macro_avg")
    cr = cr.replace("weighted avg", "weighted_avg")
    print(cr)
    lines = cr.split("\n")
    for line in lines:
        t = line.strip().split()
        if len(t) >= 5:
            classes.append(t[0])
            precision.append(t[1])
            recall.append(t[2])
            f1_score.append(t[3])
            support.append(t[4])

    return pd.DataFrame(
        data={
            "class": classes,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "support": support,
        }
    )


def plot_2d_histogram(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    xbins: int = 20,
    ybins: int = 20,
    xmax: float = np.inf,
    xmin: float = -np.inf,
    ymax: float = np.inf,
    ymin: float = -np.inf,
    z_transform: Optional[str] = None,
    z_nan: float = 0.0,
    fig=None,
):

    x = np.clip(x, xmin, xmax)
    y = np.clip(y, ymin, ymax)
    z[np.isnan(z)] = z_nan

    f = fig
    if f is None:
        f = plt.figure(figsize=(30, 20))

    a_sum = np.zeros((ybins, xbins))
    a_cnt = np.zeros_like(a_sum)

    x_min = np.min(x)
    x_max = np.max(x)
    xi = (xbins * (x - x_min - 1e-6) / (x_max - x_min)).astype(int)

    y_min = np.min(y)
    y_max = np.max(y)
    yi = (ybins * (y - y_min - 1e-6) / (y_max - y_min)).astype(int)

    if z_transform is not None:
        if z_transform == "log":
            z = np.log(z)
        elif z_transform == "sqrt":
            z = np.sqrt(z)
        elif z_transform == "sqr":
            z = z ** 2
        else:
            raise TypeError(f"z_transform {z_transform} not supported")

    for xx, yy, zz in zip(xi, yi, z):
        a_sum[yy, xx] += zz
        a_cnt[yy, xx] += 1

    a = a_sum / (a_cnt + 1e-6)
    a_min = np.min(a)
    a_max = np.max(a)
    ai = (255 * (a - a_min - 1e-6) / (a_max - a_min)).astype(int)

    aa = np.zeros((ybins, xbins, 3), dtype=int)
    aa[:, :, 0] = ai

    plt.imshow(aa, aspect="auto", origin="lower")
    xt, xl = plt.xticks()
    plt.xticks(
        xt,
        [
            (
                round(((xxtt - xt[0]) / (xt[-1] - xt[0])) * (x_max - x_min) + x_min, 2)
                if (i % 3 == 0 and i > 0)
                else ""
            )
            for i, xxtt in enumerate(xt)
        ],
    )
    xt, xl = plt.yticks()
    plt.yticks(
        xt,
        [
            (
                round(((xxtt - xt[0]) / (xt[-1] - xt[0])) * (y_max - y_min) + y_min, 2)
                if (i % 3 == 0 and i > 0)
                else ""
            )
            for i, xxtt in enumerate(xt)
        ],
    )

    return f


def df_plot_2d_histogram(
    df: pd.DataFrame,
    cx: Union[str, Any],
    cy: Union[str, Any],
    cz: Union[str, Any],
    *argc,
    **argv,
):
    f = plot_2d_histogram(
        x=df[cx].values, y=df[cy].values, z=df[cz].values, *argc, **argv
    )
    plt.xlabel(cx)
    plt.ylabel(cy)
    plt.title(f"{cz} = f({cx}, {cy})")
    return f


def example_2d_histogram():
    n = 20000
    x = np.random.random(n) + 50
    y = np.random.random(n) + 20000
    z = 1000 * x + 1000 * y ** 3 + 10 * np.random.random(n)

    # plot_2d_histogram(x=x, y=y, z=z, ybins=30)

    df = pd.DataFrame(data={"x": x, "y": y, "z": z})
    df_plot_2d_histogram(df, "x", "y", "z", xbins=100, ybins=10, ymax=20000 + 0.3)
    matplotlib.use(MATPLOTLIB_BACKEND)
    plt.show()
    matplotlib.use("Agg")


def obj_to_info(x: Any, recurse_members: Optional[Set] = None):
    if type(x) == dict:
        return x
    d = {"obj": str(type(x))}
    for attr in dir(x):
        if recurse_members and attr in recurse_members:
            d[attr] = obj_to_info(x=getattr(x, attr), recurse_members=recurse_members)
        elif attr[:2] != "__" and type(getattr(x, attr)) != types.MethodType:
            d[attr] = str(getattr(x, attr))
    return d


def create_model_inspection_report(
    model: Any,
    recurse_members: Optional[Set] = None,
    report: Optional[HtmlReport] = None,
) -> HtmlReport:
    if report is None:
        report = HtmlReport()

    info = obj_to_info(x=model, recurse_members=recurse_members)

    report.add_title(f"inspection for {model} {type(model)}")
    s = pprint.pformat(info, indent=4, width=1)
    s = s.replace("\n", "<br>").replace("    ", "&nbsp;&nbsp;&nbsp;&nbsp;")
    report.add_line(s)

    return report




if __name__ == "__main__":

    pass