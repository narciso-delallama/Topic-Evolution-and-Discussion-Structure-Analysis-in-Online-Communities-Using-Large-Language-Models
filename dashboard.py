"""
TFM Political Twitter Dashboard
===============================

Topic Evolution and Discussion Structure Analysis in Online Communities
Using Large Language Models

Author: J. Narciso de la Llama

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
import warnings
from typing import Optional

warnings.filterwarnings("ignore")

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

    def cosine_similarity_matrix(x: np.ndarray) -> np.ndarray:
        return sklearn_cosine_similarity(x)

except ImportError:

    def cosine_similarity_matrix(x: np.ndarray) -> np.ndarray:
        x = np.array(x, dtype=float)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        x_norm = x / norms
        return x_norm @ x_norm.T


# =============================================================================
# Page configuration
# =============================================================================

st.set_page_config(
    page_title="TFM Political Twitter Dashboard",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# Constants
# =============================================================================

TOPICS = [
    "Economy and Employment",
    "Welfare, Housing and Social Policy",
    "National Politics and Governance",
    "International Affairs",
    "Immigration and Security",
    "Rights and Equality",
    "Other",
]

STANCES = [
    "In favor",
    "Against",
    "Neutral",
    "Unclear",
]

ACTOR_DISPLAY = {
    "sanchezcastejon": "Pedro Sánchez",
    "NunezFeijoo": "Alberto Núñez Feijóo",
    "Santi_ABASCAL": "Santiago Abascal",
    "Yolanda_Diaz_": "Yolanda Díaz",
}

ACTOR_ORDER = [
    "Pedro Sánchez",
    "Alberto Núñez Feijóo",
    "Santiago Abascal",
    "Yolanda Díaz",
]

TOPIC_COLORS = {
    "Economy and Employment": "#4C72B0",
    "Welfare, Housing and Social Policy": "#55A868",
    "National Politics and Governance": "#C44E52",
    "International Affairs": "#8172B2",
    "Immigration and Security": "#CCB974",
    "Rights and Equality": "#64B5CD",
    "Other": "#8C8C8C",
}

STANCE_COLORS = {
    "In favor": "#4C72B0",
    "Against": "#C44E52",
    "Neutral": "#DDA85C",
    "Unclear": "#8C8C8C",
}

ACTOR_COLORS = {
    "Pedro Sánchez": "#C44E52",
    "Alberto Núñez Feijóo": "#4C72B0",
    "Santiago Abascal": "#55A868",
    "Yolanda Díaz": "#8172B2",
}

FALLBACK_PALETTE = [
    "#4C72B0",
    "#C44E52",
    "#55A868",
    "#8172B2",
    "#CCB974",
    "#64B5CD",
    "#8C8C8C",
    "#DD8452",
]

EVENTS = {
    "2022-06-19": "Andalusian election",
    "2022-06-29": "NATO Madrid Summit",
    "2022-07-14": "Democratic Memory Law",
    "2022-12-15": "Sedition reform",
    "2023-05-28": "Local and regional elections",
    "2023-07-23": "General election",
    "2023-09-29": "Failed Feijóo investiture",
    "2023-10-07": "Israel-Gaza war begins",
    "2023-11-16": "Sánchez investiture",
    "2024-02-21": "Koldo case",
    "2024-03-14": "Amnesty Law approved",
    "2024-04-24": "Sánchez reflection letter",
    "2024-10-29": "Valencia DANA",
    "2025-01-21": "Yolanda Díaz leaves X",
}

DATA_PATHS = [
    "data/classified_all_tweets_final_v2_clean.csv",
    "classified_all_tweets_final_v2_clean.csv",
    "../classified_all_tweets_final_v2_clean.csv",
    "data/classified_all_tweets_final_v2.csv",
    "classified_all_tweets_final_v2.csv",
    "../classified_all_tweets_final_v2.csv",
]


# =============================================================================
# Data utilities
# =============================================================================

def find_existing_file(paths: list[str]) -> Optional[str]:
    for path in paths:
        if os.path.exists(path):
            return path
    return None


@st.cache_data(show_spinner="Loading dataset...")
def load_main_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def standardize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Detect common column names and rename them to canonical names.

    Canonical names:
    - text
    - actor
    - actor_display
    - date
    - month
    - topic
    - stance
    - confidence
    - justification
    - url
    - tweet_id
    """

    candidates = {
        "text": ["tweet", "analysis_text", "tweet_text", "text", "full_text"],
        "actor": ["politician", "public_figure", "actor", "user"],
        "date": ["date", "datetime_utc", "datetime", "created_at", "publication_datetime"],
        "topic": ["topic", "llm_topic", "predicted_topic"],
        "stance": ["stance", "llm_stance", "predicted_stance"],
        "confidence": ["confidence", "llm_confidence", "score"],
        "justification": ["short_justification", "justification", "explanation"],
        "url": ["tweet_url", "url"],
        "tweet_id": ["tweet_id", "id"],
    }

    existing_columns = set(df.columns)
    rename_map: dict[str, str] = {}
    detected_map: dict[str, str] = {}

    for canonical, options in candidates.items():
        for option in options:
            if option in existing_columns:
                detected_map[canonical] = option
                if option != canonical:
                    rename_map[option] = canonical
                break

    df = df.rename(columns=rename_map).copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=False)
        df = df.dropna(subset=["date"]).copy()
        try:
            df["date"] = df["date"].dt.tz_localize(None)
        except TypeError:
            pass
        df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    if "actor" in df.columns:
        df["actor"] = df["actor"].astype(str)
        df["actor_display"] = df["actor"].map(ACTOR_DISPLAY).fillna(df["actor"])

    if "topic" in df.columns:
        df["topic"] = df["topic"].astype(str)
        df = df[df["topic"].isin(TOPICS)].copy()

    if "stance" in df.columns:
        df["stance"] = df["stance"].astype(str)
        df = df[df["stance"].isin(STANCES)].copy()

    if "confidence" in df.columns:
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

    return df.reset_index(drop=True), detected_map


def apply_filters(
    df: pd.DataFrame,
    actors: list[str],
    topics: list[str],
    stances: list[str],
    date_range: Optional[tuple],
    confidence_range: Optional[tuple[float, float]],
) -> pd.DataFrame:
    filtered = df.copy()

    if actors and "actor_display" in filtered.columns:
        filtered = filtered[filtered["actor_display"].isin(actors)]

    if topics and "topic" in filtered.columns:
        filtered = filtered[filtered["topic"].isin(topics)]

    if stances and "stance" in filtered.columns:
        filtered = filtered[filtered["stance"].isin(stances)]

    if date_range and "date" in filtered.columns:
        start_date = pd.Timestamp(date_range[0])
        end_date = pd.Timestamp(date_range[1])
        filtered = filtered[
            (filtered["date"] >= start_date) & (filtered["date"] <= end_date)
        ]

    if confidence_range and "confidence" in filtered.columns:
        low, high = confidence_range
        filtered = filtered[
            (filtered["confidence"] >= low) & (filtered["confidence"] <= high)
        ]

    return filtered.reset_index(drop=True)


def actor_color(actor: str, index: int = 0) -> str:
    return ACTOR_COLORS.get(actor, FALLBACK_PALETTE[index % len(FALLBACK_PALETTE)])


def get_ordered_actors(df: pd.DataFrame) -> list[str]:
    if "actor_display" not in df.columns:
        return []

    existing = sorted(df["actor_display"].dropna().unique())
    ordered = [actor for actor in ACTOR_ORDER if actor in existing]
    remaining = [actor for actor in existing if actor not in ordered]
    return ordered + remaining


def get_ordered_topics(df: pd.DataFrame) -> list[str]:
    if "topic" not in df.columns:
        return []
    return [topic for topic in TOPICS if topic in df["topic"].dropna().unique()]


def get_ordered_stances(df: pd.DataFrame) -> list[str]:
    if "stance" not in df.columns:
        return []
    return [stance for stance in STANCES if stance in df["stance"].dropna().unique()]


# =============================================================================
# Chart utilities
# =============================================================================

def make_bar_chart(
    data: pd.Series,
    title: str,
    x_label: str,
    y_label: str,
    color_map: Optional[dict[str, str]] = None,
    as_percent: bool = False,
) -> go.Figure:
    if data.empty:
        fig = go.Figure()
        fig.update_layout(title=title)
        return fig

    data = data.dropna()

    if as_percent:
        total = data.sum()
        if total > 0:
            data = data / total * 100

    colors = None
    if color_map:
        colors = [color_map.get(str(label), "#88888800") for label in data.index]

    text_values = [
        f"{value:.1f}%" if as_percent else f"{int(value):,}"
        for value in data.values
    
    ]

    fig = go.Figure(
        go.Bar(
            x=data.index.tolist(),
            y=data.values,
            marker_color=colors,
            text=text_values,
            textposition="outside",
            textfont=dict(
            color="black",
            size=12
        ),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title="%" if as_percent else y_label,
        plot_bgcolor="white",
        margin=dict(t=55, b=80, l=50, r=20),
        uniformtext_minsize=8,
        uniformtext_mode="hide",
    )

    return fig


def make_stacked_bar_chart(
    df: pd.DataFrame,
    group_col: str,
    stack_col: str,
    categories: list[str],
    color_map: dict[str, str],
    title: str,
    as_percent: bool = False,
    group_order: Optional[list[str]] = None,
) -> go.Figure:
    if group_col not in df.columns or stack_col not in df.columns or df.empty:
        fig = go.Figure()
        fig.update_layout(title=title)
        return fig

    matrix = (
        df.groupby([group_col, stack_col])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=categories, fill_value=0)
    )

    if group_order:
        ordered = [group for group in group_order if group in matrix.index]
        remaining = [group for group in matrix.index if group not in ordered]
        matrix = matrix.reindex(ordered + remaining)

    if as_percent:
        row_sums = matrix.sum(axis=1).replace(0, np.nan)
        matrix = matrix.div(row_sums, axis=0).fillna(0) * 100

    fig = go.Figure()

    for category in categories:
        if category not in matrix.columns:
            continue

        fig.add_trace(
            go.Bar(
                name=category,
                x=matrix.index.tolist(),
                y=matrix[category].tolist(),
                marker_color=color_map.get(category, "#888888"),
            )
        )

    fig.update_layout(
        title=title,
        barmode="stack",
        yaxis_title="% of tweets" if as_percent else "Tweet count",
        plot_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(t=90, b=80, l=50, r=20),
    )

    return fig


def make_heatmap(
    matrix: pd.DataFrame,
    title: str,
    colorscale: str = "Blues",
    fmt: str = ".1f",
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
    height: int = 420,
) -> go.Figure:
    if matrix.empty:
        fig = go.Figure()
        fig.update_layout(title=title)
        return fig

    text_values = [[f"{value:{fmt}}" for value in row] for row in matrix.values]

    fig = go.Figure(
        go.Heatmap(
            z=matrix.values,
            x=matrix.columns.tolist(),
            y=matrix.index.tolist(),
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            text=text_values,
            texttemplate="%{text}",
            hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>Value: %{z:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis=dict(tickangle=-30),
        margin=dict(t=65, b=100, l=150, r=20),
        height=height,
    )

    return fig


def add_event_markers(
    fig: go.Figure,
    x_min: pd.Timestamp,
    x_max: pd.Timestamp,
) -> go.Figure:
    annotation_levels = [0.96, 0.84, 0.72, 0.60]
    visible_index = 0

    for date_string, label in EVENTS.items():
        event_date = pd.Timestamp(date_string)

        if x_min <= event_date <= x_max:
            fig.add_vline(
                x=event_date,
                line=dict(color="grey", dash="dash", width=0.8),
                opacity=0.5,
            )

            fig.add_annotation(
                x=event_date,
                y=annotation_levels[visible_index % len(annotation_levels)],
                yref="paper",
                text=label,
                showarrow=False,
                font=dict(size=8, color="#555555"),
                textangle=-90,
                xanchor="right",
                align="left",
            )

            visible_index += 1

    return fig


def make_temporal_plot(
    df: pd.DataFrame,
    category_col: str,
    selected_categories: list[str],
    as_percent: bool = False,
    smooth: bool = False,
    show_events: bool = False,
    title: str = "",
    color_map: Optional[dict[str, str]] = None,
) -> go.Figure:
    if "month" not in df.columns or category_col not in df.columns or df.empty:
        fig = go.Figure()
        fig.update_layout(title=title or "No data")
        return fig

    matrix = (
        df.groupby(["month", category_col])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=selected_categories, fill_value=0)
    )

    if as_percent:
        row_sums = matrix.sum(axis=1).replace(0, np.nan)
        matrix = matrix.div(row_sums, axis=0).fillna(0) * 100

    if smooth:
        matrix = matrix.rolling(3, center=True, min_periods=1).mean()

    fig = go.Figure()

    for category in matrix.columns:
        fig.add_trace(
            go.Scatter(
                x=matrix.index.tolist(),
                y=matrix[category].round(2).tolist(),
                mode="lines",
                name=category,
                line=dict(
                    color=(color_map or {}).get(category, None),
                    width=2,
                ),
                hovertemplate=(
                    f"<b>{category}</b><br>"
                    "Month: %{x|%Y-%m}<br>"
                    "Value: %{y:.1f}<extra></extra>"
                ),
            )
        )

    if show_events and not matrix.empty:
        fig = add_event_markers(fig, matrix.index.min(), matrix.index.max())

    fig.update_layout(
        title=title,
        yaxis_title="% of monthly tweets" if as_percent else "Tweet count",
        xaxis_title="Month",
        plot_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(t=80, b=50, l=55, r=20),
    )

    return fig


def make_download_button(
    df: pd.DataFrame,
    label: str = "Download CSV",
    filename: str = "filtered_tweets.csv",
) -> None:
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label=label,
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
    )


# =============================================================================
# Graph utilities
# =============================================================================

def build_actor_topic_graph(
    df: pd.DataFrame,
    min_weight: float = 0.0,
    as_percent: bool = False,
) -> nx.Graph:
    graph = nx.Graph()

    if "actor_display" not in df.columns or "topic" not in df.columns or df.empty:
        return graph

    counts = (
        df.groupby(["actor_display", "topic"])
        .size()
        .reset_index(name="count")
    )

    if as_percent:
        totals = df.groupby("actor_display").size().reset_index(name="total")
        counts = counts.merge(totals, on="actor_display", how="left")
        counts["weight"] = counts["count"] / counts["total"] * 100
    else:
        counts["weight"] = counts["count"].astype(float)

    for _, row in counts.iterrows():
        actor = row["actor_display"]
        topic = row["topic"]
        weight = float(row["weight"])

        if weight < min_weight:
            continue

        graph.add_node(actor, node_type="actor")
        graph.add_node(topic, node_type="topic")
        graph.add_edge(actor, topic, weight=round(weight, 2))

    return graph


def build_topic_similarity_graph(
    df: pd.DataFrame,
    threshold: float = 0.70,
    by: str = "actor",
) -> tuple[nx.Graph, pd.DataFrame]:
    graph = nx.Graph()

    if "topic" not in df.columns or df.empty:
        return graph, pd.DataFrame()

    if by == "actor" and "actor_display" in df.columns:
        group_col = "actor_display"
    elif "month" in df.columns:
        group_col = "month"
    else:
        return graph, pd.DataFrame()

    matrix = (
        df.groupby([group_col, "topic"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=TOPICS, fill_value=0)
    )

    matrix = matrix.loc[:, matrix.sum() > 0]

    if matrix.shape[1] < 2:
        return graph, pd.DataFrame()

    topic_vectors = matrix.values.T.astype(float)
    similarities = cosine_similarity_matrix(topic_vectors)
    topics_present = matrix.columns.tolist()

    sim_df = pd.DataFrame(
        similarities,
        index=topics_present,
        columns=topics_present,
    ).round(3)

    for topic in topics_present:
        graph.add_node(topic, node_type="topic")

    for i, topic_a in enumerate(topics_present):
        for j, topic_b in enumerate(topics_present):
            if j <= i:
                continue

            similarity = float(similarities[i, j])

            if similarity >= threshold:
                graph.add_edge(topic_a, topic_b, weight=round(similarity, 3))

    return graph, sim_df


def plot_networkx_graph_plotly(
    graph: nx.Graph,
    title: str = "",
    node_type_attr: str = "node_type",
    show_edge_labels: bool = False,
    layout: str = "spring",
    seed: int = 42,
    height: int = 560,
) -> go.Figure:
    if len(graph.nodes) == 0:
        fig = go.Figure()
        fig.update_layout(title="No data to display")
        return fig

    if layout == "bipartite":
        actor_nodes = [
            node
            for node, attrs in graph.nodes(data=True)
            if attrs.get(node_type_attr) == "actor"
        ]

        try:
            positions = nx.bipartite_layout(graph, actor_nodes, scale=2.0)
        except Exception:
            positions = nx.spring_layout(graph, seed=seed, k=1.8)

    else:
        positions = nx.spring_layout(graph, seed=seed, k=1.8)

    traces = []
    weights = [attrs.get("weight", 1.0) for _, _, attrs in graph.edges(data=True)]
    max_weight = max(weights, default=1.0)

    for node_a, node_b, attrs in graph.edges(data=True):
        x0, y0 = positions[node_a]
        x1, y1 = positions[node_b]
        weight = attrs.get("weight", 1.0)
        width = 0.5 + 5.5 * (weight / max_weight) if max_weight > 0 else 1.5

        traces.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=width, color="#BBBBBB"),
                hoverinfo="text",
                hovertext=f"{node_a} — {node_b}: {weight:.2f}",
                showlegend=False,
            )
        )

        if show_edge_labels:
            traces.append(
                go.Scatter(
                    x=[(x0 + x1) / 2],
                    y=[(y0 + y1) / 2],
                    mode="text",
                    text=[f"{weight:.2f}"],
                    textfont=dict(size=8, color="#555555"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    node_x = []
    node_y = []
    node_text = []
    node_hover = []
    node_color = []
    node_size = []

    for node, attrs in graph.nodes(data=True):
        x, y = positions[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(str(node))

        node_type = attrs.get(node_type_attr, "topic")

        if node_type == "actor":
            node_color.append(ACTOR_COLORS.get(node, "#C44E52"))
            node_size.append(30)
        else:
            node_color.append(TOPIC_COLORS.get(node, "#888888"))
            node_size.append(22)

        degree = graph.degree(node)
        total_weight = sum(
            edge_attrs.get("weight", 1.0)
            for _, _, edge_attrs in graph.edges(node, data=True)
        )

        node_hover.append(
            f"<b>{node}</b><br>"
            f"Edges: {degree}<br>"
            f"Total weight: {total_weight:.2f}"
        )

    traces.append(
    go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        textfont=dict(
            size=11,
            color="#222222",
            family="Arial"
        ),
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=1.5, color="#333333"),
        ),
        hovertext=node_hover,
        hoverinfo="text",
        showlegend=False,
    )
)

    fig = go.Figure(data=traces)

    fig.update_layout(
        title=title,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white",
        height=height,
        margin=dict(t=60, b=20, l=20, r=20),
    )

    return fig


# =============================================================================
# Tab renderers
# =============================================================================

def render_tab_overview(df: pd.DataFrame, as_percent: bool) -> None:
    st.header("Corpus Overview")
    st.markdown(
        "This tab summarizes the filtered corpus. Use the sidebar to filter by "
        "actor, topic, stance, date range and confidence."
    )

    has_confidence = "confidence" in df.columns

    metric_columns = st.columns(5 if has_confidence else 4)

    metric_columns[0].metric("Tweets", f"{len(df):,}")

    if "actor_display" in df.columns:
        metric_columns[1].metric("Actors", df["actor_display"].nunique())
    else:
        metric_columns[1].metric("Actors", "—")

    if "date" in df.columns and len(df) > 0:
        min_date = df["date"].min().strftime("%d %b %Y")
        max_date = df["date"].max().strftime("%d %b %Y")
        metric_columns[2].metric("Date range", f"{min_date} → {max_date}")
    else:
        metric_columns[2].metric("Date range", "—")

    if "topic" in df.columns:
        metric_columns[3].metric("Topics", df["topic"].nunique())
    else:
        metric_columns[3].metric("Topics", "—")

    if has_confidence:
        metric_columns[4].metric("Mean confidence", f"{df['confidence'].mean():.3f}")

    st.divider()

    if "actor_display" in df.columns and len(df) > 0:
        counts = df["actor_display"].value_counts()
        color_map = {
            actor: actor_color(actor, i)
            for i, actor in enumerate(counts.index)
        }

        fig = make_bar_chart(
            counts,
            title="Tweet count by actor",
            x_label="Actor",
            y_label="Tweets",
            color_map=color_map,
            as_percent=as_percent,
        )
        st.plotly_chart(fig, width="stretch", key="overview_actor_count")

    if "month" in df.columns and len(df) > 0:
        monthly = df.groupby("month").size().reset_index(name="count")
        fig = px.line(
            monthly,
            x="month",
            y="count",
            title="Monthly tweet volume",
            labels={"month": "Month", "count": "Tweets"},
        )
        fig.update_traces(line_color="#4C72B0", line_width=2)
        fig.update_layout(plot_bgcolor="white", margin=dict(t=55, b=50))
        st.plotly_chart(fig, width="stretch", key="overview_monthly_volume")

    if "topic" in df.columns and len(df) > 0:
        topic_counts = df["topic"].value_counts().reindex(TOPICS).dropna()

        fig = make_bar_chart(
            topic_counts,
            title="Topic distribution",
            x_label="Topic",
            y_label="Tweets",
            color_map=TOPIC_COLORS,
            as_percent=as_percent,
        )
        fig.update_xaxes(tickangle=-25)
        st.plotly_chart(fig, width="stretch", key="overview_topic_distribution")

    if "stance" in df.columns and len(df) > 0:
        stance_counts = df["stance"].value_counts().reindex(STANCES).dropna()

        fig = make_bar_chart(
            stance_counts,
            title="Stance distribution",
            x_label="Stance",
            y_label="Tweets",
            color_map=STANCE_COLORS,
            as_percent=as_percent,
        )
        st.plotly_chart(fig, width="stretch", key="overview_stance_distribution")


def render_tab_distributions(df: pd.DataFrame, as_percent: bool) -> None:
    st.header("Topic and Stance Distributions")
    st.markdown(
        "This tab shows global and actor-level distributions. "
        "Use the sidebar toggle to switch between counts and percentages."
    )

    if "topic" in df.columns and len(df) > 0:
        topic_counts = df["topic"].value_counts().reindex(TOPICS).dropna()

        fig = make_bar_chart(
            topic_counts,
            title="Topic distribution",
            x_label="Topic",
            y_label="Tweets",
            color_map=TOPIC_COLORS,
            as_percent=as_percent,
        )
        fig.update_xaxes(tickangle=-25)
        st.plotly_chart(fig, width="stretch", key="dist_topic_global")

    if "stance" in df.columns and len(df) > 0:
        stance_counts = df["stance"].value_counts().reindex(STANCES).dropna()

        fig = make_bar_chart(
            stance_counts,
            title="Stance distribution",
            x_label="Stance",
            y_label="Tweets",
            color_map=STANCE_COLORS,
            as_percent=as_percent,
        )
        st.plotly_chart(fig, width="stretch", key="dist_stance_global")

    st.divider()
    st.subheader("Distributions by actor")

    if "actor_display" in df.columns and "topic" in df.columns and len(df) > 0:
        fig = make_stacked_bar_chart(
            df=df,
            group_col="actor_display",
            stack_col="topic",
            categories=TOPICS,
            color_map=TOPIC_COLORS,
            title="Topic distribution by actor",
            as_percent=as_percent,
            group_order=ACTOR_ORDER,
        )
        st.plotly_chart(fig, width="stretch", key="dist_topic_by_actor")

    if "actor_display" in df.columns and "stance" in df.columns and len(df) > 0:
        fig = make_stacked_bar_chart(
            df=df,
            group_col="actor_display",
            stack_col="stance",
            categories=STANCES,
            color_map=STANCE_COLORS,
            title="Stance distribution by actor",
            as_percent=as_percent,
            group_order=ACTOR_ORDER,
        )
        st.plotly_chart(fig, width="stretch", key="dist_stance_by_actor")


def render_tab_actor_comparison(df: pd.DataFrame) -> None:
    st.header("Actor Comparison")
    st.markdown(
        "The following matrices are normalized by actor. Each row sums to 100%, "
        "so actors with different tweet volumes can be compared."
    )

    if "actor_display" not in df.columns or len(df) == 0:
        st.info("Actor column not found or no data after filtering.")
        return

    if "topic" in df.columns:
        topic_matrix = (
            pd.crosstab(df["actor_display"], df["topic"], normalize="index") * 100
        ).reindex(columns=TOPICS, fill_value=0).round(1)

        ordered = [actor for actor in ACTOR_ORDER if actor in topic_matrix.index]
        remaining = [actor for actor in topic_matrix.index if actor not in ordered]
        topic_matrix = topic_matrix.reindex(ordered + remaining)

        fig = make_heatmap(
            topic_matrix,
            title="Topic share by actor (%)",
            colorscale="Blues",
            fmt=".1f",
        )
        st.plotly_chart(fig, width="stretch", key="actor_topic_heatmap")
        st.caption("Each cell shows the percentage of that actor's tweets assigned to a given topic.")

    if "stance" in df.columns:
        stance_matrix = (
            pd.crosstab(df["actor_display"], df["stance"], normalize="index") * 100
        ).reindex(columns=STANCES, fill_value=0).round(1)

        ordered = [actor for actor in ACTOR_ORDER if actor in stance_matrix.index]
        remaining = [actor for actor in stance_matrix.index if actor not in ordered]
        stance_matrix = stance_matrix.reindex(ordered + remaining)

        fig = make_heatmap(
            stance_matrix,
            title="Stance share by actor (%)",
            colorscale="Oranges",
            fmt=".1f",
        )
        st.plotly_chart(fig, width="stretch", key="actor_stance_heatmap")
        st.caption("Each cell shows the percentage of that actor's tweets assigned to a given stance.")

    st.divider()
    st.subheader("Agenda similarity between actors")
    st.markdown(
        "Cosine similarity is computed using the topic distribution vector of each actor. "
        "A value close to 1 means that two actors have similar topic proportions."
    )

    if "topic" in df.columns and df["actor_display"].nunique() >= 2:
        actor_vectors = (
            pd.crosstab(df["actor_display"], df["topic"], normalize="index")
        ).reindex(columns=TOPICS, fill_value=0)

        ordered = [actor for actor in ACTOR_ORDER if actor in actor_vectors.index]
        remaining = [actor for actor in actor_vectors.index if actor not in ordered]
        actor_vectors = actor_vectors.reindex(ordered + remaining)

        similarity = cosine_similarity_matrix(actor_vectors.values)
        similarity_df = pd.DataFrame(
            similarity,
            index=actor_vectors.index,
            columns=actor_vectors.index,
        ).round(3)

        fig = make_heatmap(
            similarity_df,
            title="Actor-actor agenda similarity",
            colorscale="Greens",
            fmt=".3f",
            zmin=0,
            zmax=1,
            height=420,
        )
        st.plotly_chart(fig, width="stretch", key="actor_similarity_heatmap")


def render_tab_temporal(df: pd.DataFrame, as_percent: bool) -> None:
    st.header("Temporal Evolution")
    st.markdown(
        "This tab shows the monthly evolution of topic or stance activity. "
        "The plots can be smoothed with a three-month rolling average."
    )

    if "month" not in df.columns or len(df) == 0:
        st.info("Date column not found or no data after filtering.")
        return

    col_view, col_actor, col_smooth, col_events = st.columns([1.3, 3, 1, 1])

    view = col_view.radio(
        "View by",
        ["Topic", "Stance"],
        horizontal=True,
        key="temporal_view_radio",
    )

    available_actors = get_ordered_actors(df)
    selected_actors = col_actor.multiselect(
        "Actors",
        available_actors,
        default=available_actors,
        key="temporal_actor_selector",
    )

    smooth = col_smooth.checkbox(
        "Smooth",
        value=True,
        key="temporal_smooth_checkbox",
    )

    show_events = col_events.checkbox(
        "Events",
        value=False,
        key="temporal_events_checkbox",
    )

    if view == "Topic":
        category_col = "topic"
        category_pool = get_ordered_topics(df)
        color_map = TOPIC_COLORS
    else:
        category_col = "stance"
        category_pool = get_ordered_stances(df)
        color_map = STANCE_COLORS

    selected_categories = st.multiselect(
        f"Select {view.lower()}s to display",
        category_pool,
        default=category_pool,
        key=f"temporal_category_selector_{view}",
    )

    if not selected_categories:
        st.info("Select at least one category.")
        return

    temporal_df = df.copy()

    if selected_actors and "actor_display" in temporal_df.columns:
        temporal_df = temporal_df[temporal_df["actor_display"].isin(selected_actors)]

    if temporal_df.empty:
        st.warning("No data for the selected filters.")
        return

    actors_to_plot = [
        actor
        for actor in selected_actors
        if "actor_display" in temporal_df.columns
        and actor in temporal_df["actor_display"].unique()
    ]

    if len(actors_to_plot) == 0:
        fig = make_temporal_plot(
            temporal_df,
            category_col=category_col,
            selected_categories=selected_categories,
            as_percent=as_percent,
            smooth=smooth,
            show_events=show_events,
            title=f"Monthly {view.lower()} evolution",
            color_map=color_map,
        )
        st.plotly_chart(fig, width="stretch", key="temporal_all_actors")

    elif len(actors_to_plot) == 1:
        actor = actors_to_plot[0]
        actor_df = temporal_df[temporal_df["actor_display"] == actor]

        fig = make_temporal_plot(
            actor_df,
            category_col=category_col,
            selected_categories=selected_categories,
            as_percent=as_percent,
            smooth=smooth,
            show_events=show_events,
            title=f"{actor}: monthly {view.lower()} evolution",
            color_map=color_map,
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, width="stretch", key=f"temporal_single_{view}_{actor}")

    else:
        for index, actor in enumerate(actors_to_plot):
            actor_df = temporal_df[temporal_df["actor_display"] == actor]

            fig = make_temporal_plot(
                actor_df,
                category_col=category_col,
                selected_categories=selected_categories,
                as_percent=as_percent,
                smooth=smooth,
                show_events=show_events,
                title=actor,
                color_map=color_map,
            )

            fig.update_layout(
                height=430,
                showlegend=True,
            )

            safe_actor_key = actor.replace(" ", "_").replace("á", "a").replace("é", "e")
            st.plotly_chart(
                fig,
                width="stretch",
                key=f"temporal_vertical_{view}_{safe_actor_key}_{index}",
            )


def render_tab_graphs(df: pd.DataFrame, as_percent: bool) -> None:
    st.header("Thematic Graphs")
    st.info(
        "These graphs are interpretability tools. They show structural patterns "
        "in the classified corpus, but they do not imply causal relationships."
    )

    tab_actor_topic, tab_topic_topic = st.tabs(
        ["Actor-topic bipartite graph", "Topic-topic similarity graph"]
    )

    with tab_actor_topic:
        st.subheader("Actor-topic bipartite graph")
        st.markdown(
            "Actors and topics are represented as nodes. Edge thickness is proportional "
            "to tweet count or actor-normalized percentage."
        )

        col_min, col_layout, col_seed, col_labels = st.columns(4)

        min_weight = col_min.slider(
            "Minimum edge weight",
            0.0,
            100.0,
            2.0,
            1.0,
            key="actor_topic_min_weight",
        )

        layout = col_layout.selectbox(
            "Layout",
            ["spring", "bipartite"],
            key="actor_topic_layout",
        )

        seed = col_seed.number_input(
            "Layout seed",
            min_value=0,
            max_value=9999,
            value=42,
            key="actor_topic_seed",
        )

        show_edge_labels = col_labels.checkbox(
            "Edge labels",
            value=False,
            key="actor_topic_edge_labels",
        )

        graph = build_actor_topic_graph(
            df,
            min_weight=min_weight,
            as_percent=as_percent,
        )

        if len(graph.edges) == 0:
            st.warning("No edges above the selected threshold. Try lowering the minimum edge weight.")
        else:
            fig = plot_networkx_graph_plotly(
                graph,
                title="Actor-topic bipartite graph",
                show_edge_labels=show_edge_labels,
                layout=layout,
                seed=int(seed),
            )
            st.plotly_chart(fig, width="stretch", key="actor_topic_graph")

    with tab_topic_topic:
        st.subheader("Topic-topic similarity graph")
        st.markdown(
            "An edge connects two topics when their distributions across actors or months "
            "are similar. Edge presence reflects similarity in distribution, not causal influence."
        )

        col_threshold, col_by, col_seed, col_labels = st.columns(4)

        threshold = col_threshold.slider(
            "Similarity threshold",
            0.0,
            1.0,
            0.70,
            0.05,
            key="topic_similarity_threshold",
        )

        by = col_by.radio(
            "Vectors by",
            ["actor", "month"],
            key="topic_similarity_by",
        )

        seed = col_seed.number_input(
            "Layout seed",
            min_value=0,
            max_value=9999,
            value=42,
            key="topic_similarity_seed",
        )

        show_edge_labels = col_labels.checkbox(
            "Edge labels",
            value=False,
            key="topic_similarity_edge_labels",
        )

        graph, similarity_df = build_topic_similarity_graph(
            df,
            threshold=threshold,
            by=by,
        )

        if not similarity_df.empty:
            with st.expander("Similarity matrix"):
                st.dataframe(
                    similarity_df.style.background_gradient(
                        cmap="Greens",
                        vmin=0,
                        vmax=1,
                    ).format("{:.3f}"),
                    width="stretch",
                )

        if len(graph.edges) == 0:
            st.warning("No topic pairs above the selected threshold. Try lowering it.")
        else:
            fig = plot_networkx_graph_plotly(
                graph,
                title="Topic-topic similarity graph",
                show_edge_labels=show_edge_labels,
                seed=int(seed),
            )
            st.plotly_chart(fig, width="stretch", key="topic_similarity_graph")


def render_tab_explorer(df: pd.DataFrame) -> None:
    st.header("Tweet Explorer")
    st.markdown("Browse individual tweets and inspect their assigned topic and stance labels.")

    explorer_df = df.copy()

    keyword = st.text_input(
        "Keyword search",
        placeholder="Filter by keyword...",
        key="explorer_keyword",
    )

    if keyword and "text" in explorer_df.columns:
        explorer_df = explorer_df[
            explorer_df["text"].str.contains(keyword, case=False, na=False)
        ]

    rows_to_show = st.select_slider(
        "Rows to display",
        options=[25, 50, 100, 200, 500],
        value=100,
        key="explorer_rows_to_show",
    )

    column_order = [
        "date",
        "actor_display",
        "text",
        "topic",
        "stance",
        "confidence",
        "justification",
        "url",
        "tweet_id",
    ]

    display_columns = [col for col in column_order if col in explorer_df.columns]

    if not display_columns:
        st.info("No displayable columns found.")
        return

    display_df = explorer_df[display_columns].head(rows_to_show).copy()

    display_df = display_df.rename(
        columns={
            "actor_display": "actor",
            "text": "tweet",
            "justification": "short_justification",
        }
    )

    if "date" in display_df.columns:
        display_df["date"] = pd.to_datetime(display_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    st.markdown(
        f"Showing **{len(display_df):,}** tweets "
        f"out of **{len(explorer_df):,}** tweets matching the filters."
    )

    if "url" in display_df.columns:
        html_df = display_df.copy()
        html_df["url"] = html_df["url"].apply(
            lambda url: f'<a href="{url}" target="_blank">link</a>'
            if pd.notna(url) and str(url).strip()
            else ""
        )

        st.write(
            html_df.to_html(escape=False, index=False, border=0),
            unsafe_allow_html=True,
        )
    else:
        st.dataframe(display_df, width="stretch", hide_index=True)

    st.divider()

    download_df = explorer_df[display_columns].rename(
        columns={
            "actor_display": "actor",
            "text": "tweet",
            "justification": "short_justification",
        }
    )

    make_download_button(
        download_df,
        label="Download filtered tweets as CSV",
        filename="filtered_tweets.csv",
    )


def render_tab_about() -> None:
    st.header("About / Methodology")

    st.markdown(
        """
## Project

**Topic Evolution and Discussion Structure Analysis in Online Communities Using Large Language Models**

Master's Thesis, J. Narciso de la Llama

---

## Pipeline

```

Scraping from X/Twitter
↓
Raw tweet storage
↓
Preprocessing and cleaning
↓
LLM-ready corpus construction
↓
Prompt testing and refinement
↓
LLM topic and stance classification
↓
Gold standard human validation
↓
Traditional NLP comparison
↓
Temporal topic evolution analysis
↓
Thematic graph construction
↓
Dashboard

```

---

## Corpus

| Handle | Name |
|---|---|
| `sanchezcastejon` | Pedro Sánchez |
| `NunezFeijoo` | Alberto Núñez Feijóo |
| `Santi_ABASCAL` | Santiago Abascal |
| `Yolanda_Diaz_` | Yolanda Díaz |

**Observation window:** 23 April 2022 to 23 April 2026.

---

## Classification schema

**Topics:**

1. Economy and Employment  
2. Welfare, Housing and Social Policy  
3. National Politics and Governance  
4. International Affairs  
5. Immigration and Security  
6. Rights and Equality  
7. Other  

**Stances:**

1. In favor  
2. Against  
3. Neutral  
4. Unclear  

---

## LLM classification

Classification was performed with **Gemini `gemini-3.1-flash-lite-preview`**.

Each tweet was assigned:

| Field | Description |
|---|---|
| `topic` | One of the seven topic categories |
| `stance` | One of the four stance labels |
| `confidence` | Self-reported model confidence between 0 and 1 |
| `short_justification` | Short explanation of the classification |

**Important:** confidence is auxiliary metadata. It should not be interpreted as proof of correctness.

---

## Dashboard scope

This dashboard does not scrape tweets, preprocess data, call the Gemini API, or evaluate the model.
It only visualizes the final classified corpus.

The dashboard is an exploratory and interpretability tool, not a prediction system.
The thematic graphs show structural patterns in the classified corpus, not causal relationships.
"""
    )


# =============================================================================
# Main app
# =============================================================================

def main() -> None:
    st.title("TFM Political Twitter Dashboard")
    st.markdown(
        "_Topic Evolution and Discussion Structure in Spanish Political Twitter, "
        "Master's Thesis, J. Narciso de la Llama_"
    )
    st.divider()

    data_path = find_existing_file(DATA_PATHS)

    if data_path is None:
        st.error(
            "Dataset not found.\n\n"
            "Please place the final classified corpus at:\n\n"
            "`data/classified_all_tweets_final_v2_clean.csv`\n\n"
            "or in the same folder as `app.py` with the name:\n\n"
            "`classified_all_tweets_final_v2_clean.csv`"
        )
        st.stop()

    raw_df = load_main_data(data_path)
    df, detected_columns = standardize_columns(raw_df)

    if df.empty:
        st.error("The dataset was loaded, but it contains no usable rows.")
        st.stop()

    with st.sidebar:
        st.header("Filters")

        actors_all = get_ordered_actors(df)
        selected_actors = st.multiselect(
            "Actor",
            actors_all,
            default=actors_all,
            key="sidebar_actor_filter",
        )

        topics_all = get_ordered_topics(df)
        selected_topics = st.multiselect(
            "Topic",
            topics_all,
            default=topics_all,
            key="sidebar_topic_filter",
        )

        stances_all = get_ordered_stances(df)
        selected_stances = st.multiselect(
            "Stance",
            stances_all,
            default=stances_all,
            key="sidebar_stance_filter",
        )

        date_range = None

        if "date" in df.columns and len(df) > 0:
            min_date = df["date"].min().date()
            max_date = df["date"].max().date()

            selected_date_range = st.date_input(
                "Date range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                key="sidebar_date_range",
            )

            if isinstance(selected_date_range, (list, tuple)) and len(selected_date_range) == 2:
                date_range = selected_date_range
            else:
                date_range = (min_date, max_date)

        confidence_range = None

        if "confidence" in df.columns:
            confidence_range = st.slider(
                "Confidence",
                min_value=0.0,
                max_value=1.0,
                value=(0.0, 1.0),
                step=0.05,
                key="sidebar_confidence_range",
            )

        as_percent = st.toggle(
            "Show as percentage",
            value=False,
            key="sidebar_percentage_toggle",
        )

        st.divider()
        st.caption(f"Loaded file: `{os.path.basename(data_path)}`")
        st.caption(f"Rows loaded: **{len(df):,}**")

    filtered_df = apply_filters(
        df=df,
        actors=selected_actors,
        topics=selected_topics,
        stances=selected_stances,
        date_range=date_range,
        confidence_range=confidence_range,
    )

    if filtered_df.empty:
        st.warning(
            "No tweets match the current filter combination. "
            "Please broaden the selection in the sidebar."
        )

    tab_overview, tab_distributions, tab_actor, tab_temporal, tab_graphs, tab_explorer, tab_about = st.tabs(
        [
            "Corpus Overview",
            "Distributions",
            "Actor Comparison",
            "Temporal Evolution",
            "Thematic Graphs",
            "Tweet Explorer",
            "About",
        ]
    )

    with tab_overview:
        render_tab_overview(filtered_df, as_percent)

    with tab_distributions:
        render_tab_distributions(filtered_df, as_percent)

    with tab_actor:
        render_tab_actor_comparison(filtered_df)

    with tab_temporal:
        render_tab_temporal(filtered_df, as_percent)

    with tab_graphs:
        render_tab_graphs(filtered_df, as_percent)

    with tab_explorer:
        render_tab_explorer(filtered_df)

    with tab_about:
        render_tab_about()


if __name__ == "__main__":
    main()

