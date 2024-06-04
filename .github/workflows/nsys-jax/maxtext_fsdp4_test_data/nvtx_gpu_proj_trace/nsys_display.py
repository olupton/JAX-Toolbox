# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import platform

import plotly.express as px
import plotly.graph_objects as go
from ipywidgets import Dropdown, interact
from plotly.subplots import make_subplots


def get_stats_cols(df):
    column_names_set = set(df.columns)
    q1 = "Q1" if "Q1" in column_names_set else "Q1 (approx)"
    q3 = "Q3" if "Q3" in column_names_set else "Q3 (approx)"

    if "Med" in column_names_set:
        med = "Med"
    elif "Median" in column_names_set:
        med = "Median"
    else:
        med = "Median (approx)"

    if "StdDev" in column_names_set:
        std = "StdDev"
    elif "Std" in column_names_set:
        std = "Std"
    else:
        std = "Std (approx)"

    return q1, med, q3, std


def display_box(df, x=None, **layout_args):
    if x is None:
        x = df.index

    q1, med, q3, std = get_stats_cols(df)

    fig = go.Figure()
    fig.add_trace(
        go.Box(
            x=x,
            q1=df[q1],
            median=df[med],
            q3=df[q3],
            lowerfence=df["Min"],
            upperfence=df["Max"],
            sd=df[std],
        )
    )

    fig.update_layout(**layout_args)
    fig.show()


def display_stats_scatter(df, x=None, **layout_args):
    if x is None:
        x = df.index

    fig = go.Figure()

    q1, med, q3, _ = get_stats_cols(df)
    col_names = [q1, med, q3, "Min", "Max"]

    for name in col_names:
        fig.add_trace(go.Scatter(x=x, y=df[name], name=name))

    fig.update_layout(**layout_args)
    fig.show()


def display_table_per_rank(df):
    if df.empty:
        display(df)
        return

    rank_groups = df.groupby("Rank")

    def display_table(name):
        rank_df = rank_groups.get_group(name)
        rank_df = rank_df.drop(columns=["Rank"])
        display(rank_df)

    dropdown = Dropdown(
        options=rank_groups.groups.keys(),
        layout={"width": "max-content"},
        description="rank",
    )

    interact(display_table, name=dropdown)


def display_stats_per_operation(
    df, x=None, box=True, scatter=True, table=True, **layout_args
):
    if df.empty:
        display(df)
        return

    if x is None:
        x = df.index

    op_groups = df.groupby(x)

    def display_graphs(name):
        op_df = op_groups.get_group(name)
        if table:
            display(op_df.reset_index(drop=True).set_index("Rank"))
        if box:
            display_box(op_df, x=op_df["Rank"], **layout_args)
        if scatter:
            display_stats_scatter(op_df, x=op_df["Rank"], **layout_args)

    operations = list(op_groups.groups.keys())

    # Plot is not being displayed for the default dropdown value. If there is
    # only one element, do not create the dropdown. Otherwise, set it to the
    # second element before resetting to the first one.
    if len(operations) > 1:
        dropdown = Dropdown(
            options=operations, layout={"width": "max-content"}, value=operations[1]
        )
        interact(display_graphs, name=dropdown)
        dropdown.value = operations[0]
    elif len(operations) == 1:
        display_graphs(operations[0])


def display_summary_graph(df, value_col, **layout_args):
    # For px.line(), 'svg' is used for figures with less than 1000 data points
    # and 'webgl' is used for larger figures. However, 'webgl' isn't supported
    # by nsys-ui on Windows, so we use 'svg' in all cases.
    if platform.system() == "Windows":
        render_mode = "svg"
    else:
        render_mode = "auto"

    fig = px.line(df.groupby("Duration")[value_col].mean(), render_mode=render_mode)
    fig.update_layout(**layout_args)
    fig.show()


def _get_heatmap_height(name_count, plot_count=1):
    name_count = max(name_count, 9)
    return (name_count * 27 + 110) * plot_count


def display_heatmaps(
    df, types, xaxis_title, yaxis_title, zaxis_title, zmax=100, **layout_args
):
    unique_name_count = df["Name"].nunique()
    height = _get_heatmap_height(unique_name_count, len(types))

    fig = make_subplots(
        len(types), 1, subplot_titles=types, vertical_spacing=150 / height
    )

    for index, type in enumerate(types):
        fig.add_trace(
            go.Heatmap(
                x=df["Duration"],
                y=df["Name"],
                z=df[type],
                showscale=False,
                zmax=zmax,
                zauto=False,
            ),
            index + 1,
            1,
        )

    fig.update_layout(height=height, **layout_args)
    fig.update_xaxes(title=xaxis_title)
    fig.update_yaxes(
        title=yaxis_title, categoryorder="category descending", nticks=unique_name_count
    )
    fig.update_traces({"colorbar": {"title_text": zaxis_title}}, showscale=True, row=0)
    fig.update_traces(
        hovertemplate=f"{xaxis_title}: %{{x}}<br>{yaxis_title}: %{{y}}<br>{zaxis_title}: %{{z}}<extra></extra>"
    )
    fig.show()


def display_heatmap(
    df, value_col, xaxis_title, yaxis_title, zaxis_title, zmax=100, **layout_args
):
    fig = px.imshow(
        df.pivot(index="Name", columns="Duration")[value_col], range_color=(0, zmax)
    )

    unique_name_count = df["Name"].nunique()
    fig.update_layout(
        height=_get_heatmap_height(unique_name_count),
        coloraxis_colorbar_title=zaxis_title,
        **layout_args,
    )
    fig.update_xaxes(title=xaxis_title)
    fig.update_yaxes(title=yaxis_title, nticks=unique_name_count)
    fig.update_traces(
        hovertemplate=f"{xaxis_title}: %{{x}}<br>{yaxis_title}: %{{y}}<br>{zaxis_title}: %{{z}}<extra></extra>"
    )

    fig.show()
