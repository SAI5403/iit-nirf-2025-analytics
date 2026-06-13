from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="IIT NIRF 2025 Analytics",
    page_icon="📊",
    layout="wide",
)


ROOT = Path(__file__).resolve().parent
DATA_DIR_CANDIDATES = [
    ROOT / "data",
    ROOT,
    ROOT.parent / "data",
]


def find_data_file(filename: str) -> Path:
    for directory in DATA_DIR_CANDIDATES:
        path = directory / filename
        if path.exists():
            return path
    return DATA_DIR_CANDIDATES[0] / filename


MASTER_FILE = find_data_file("00_MASTER_IIT_NIRF_2025.csv")
AGE_FILE = find_data_file("09_faculty_age_group_summary.csv")
JOIN_FILE = find_data_file("10_faculty_joining_period_summary.csv")

CORE_FEATURES = [
    "nirf_score",
    "tlr_score",
    "rpc_score",
    "go_score",
    "oi_score",
    "perception_score",
    "total_faculty",
    "faculty_with_phd",
    "total_students",
    "placement_rate_pct",
    "ug_median_salary_lakh",
    "phd_pct",
    "phd_output_rate_pct",
    "patents_published",
    "patents_granted",
    "sponsored_funding_cr",
    "consultancy_revenue_cr",
    "research_intensity_cr",
    "lab_capex_cr",
    "salary_expense_cr",
    "institution_age",
]


@st.cache_data
def load_data():
    if not MASTER_FILE.exists():
        st.error(
            "Could not find 00_MASTER_IIT_NIRF_2025.csv. "
            "Keep the CSV files inside the data/ folder."
        )
        st.stop()

    df = pd.read_csv(MASTER_FILE)
    text_cols = {"institute_id", "short_name", "full_name", "city", "state", "generation"}
    for col in df.columns:
        if col not in text_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    age_df = pd.read_csv(AGE_FILE) if AGE_FILE.exists() else pd.DataFrame()
    join_df = pd.read_csv(JOIN_FILE) if JOIN_FILE.exists() else pd.DataFrame()
    return df, age_df, join_df


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=np.number).columns.tolist()


def feature_matrix(df: pd.DataFrame, features: list[str] | None = None):
    chosen = [c for c in (features or CORE_FEATURES) if c in df.columns]
    x = df[chosen].replace([np.inf, -np.inf], np.nan)
    return x.fillna(x.median(numeric_only=True)), chosen


def pca_projection(df: pd.DataFrame, features: list[str] | None = None):
    x, cols = feature_matrix(df, features)
    scaled = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(scaled)
    result = df[["short_name", "nirf_rank", "generation"]].copy()
    result["PC1"] = coords[:, 0]
    result["PC2"] = coords[:, 1]
    loadings = pd.DataFrame(pca.components_.T, columns=["PC1", "PC2"], index=cols)
    return result, loadings, pca.explained_variance_ratio_, scaled, cols


def clustered_data(df: pd.DataFrame, k: int = 3, features: list[str] | None = None):
    pca_df, _, _, scaled, cols = pca_projection(df, features)
    labels = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(scaled)
    out = df.copy()
    out["cluster"] = labels
    out["cluster_label"] = out["cluster"].map(lambda value: f"Cluster {value + 1}")
    pca_df["cluster_label"] = out["cluster_label"]
    return out, pca_df, scaled, cols


def normalized_scores(df: pd.DataFrame, columns: list[str]):
    norm = pd.DataFrame(index=df.index)
    for col in columns:
        series = pd.to_numeric(df[col], errors="coerce")
        low, high = series.min(), series.max()
        norm[col] = 0.5 if high == low else (series - low) / (high - low)
    return norm.fillna(0)


df, age_df, join_df = load_data()
all_numeric = numeric_columns(df)
clustered_all, _, _, _ = clustered_data(df, 3)

with st.sidebar:
    st.title("IIT NIRF 2025")
    generations = sorted(df["generation"].dropna().unique()) if "generation" in df else []
    selected_generations = st.multiselect("Generation", generations, default=generations)
    rank_max = int(df["nirf_rank"].max())
    rank_range = st.slider("Rank range", 1, rank_max, (1, rank_max))
    color_options = [c for c in ["generation", "state"] if c in df.columns] + ["cluster_label"]
    color_by = st.selectbox("Color by", color_options)

filtered = df.copy()
if selected_generations:
    filtered = filtered[filtered["generation"].isin(selected_generations)]
filtered = filtered[
    (filtered["nirf_rank"] >= rank_range[0]) & (filtered["nirf_rank"] <= rank_range[1])
]
filtered = filtered.merge(
    clustered_all[["institute_id", "cluster_label"]], on="institute_id", how="left"
)

st.title("IIT NIRF 2025 Interactive Analytics Dashboard")
st.caption(
    "Dashboard layer for IIT comparison, EDA, faculty analysis, correlations, "
    "PCA, KMeans clustering, and ranking sensitivity simulation."
)

metric_cols = st.columns(4)
metric_cols[0].metric("IITs in view", len(filtered))
metric_cols[1].metric("Best rank", int(filtered["nirf_rank"].min()) if len(filtered) else "-")
metric_cols[2].metric(
    "Average NIRF score", f"{filtered['nirf_score'].mean():.2f}" if len(filtered) else "-"
)
metric_cols[3].metric(
    "Average placement",
    f"{filtered['placement_rate_pct'].mean():.1f}%" if len(filtered) else "-",
)

tabs = st.tabs(
    [
        "IIT Comparison",
        "Trends & EDA",
        "Faculty Analysis",
        "Correlation Explorer",
        "PCA & Clustering",
        "Ranking Simulator",
        "Audit & Insights",
        "Data",
    ]
)

with tabs[0]:
    st.subheader("Side-by-side institutional comparison")
    default_iits = df.sort_values("nirf_rank")["short_name"].head(5).tolist()
    chosen_iits = st.multiselect("Select IITs", df["short_name"].tolist(), default=default_iits)
    comparison = df[df["short_name"].isin(chosen_iits)].sort_values("nirf_rank")
    available_metrics = [c for c in CORE_FEATURES if c in df.columns]
    selected_metrics = st.multiselect(
        "Metrics",
        available_metrics,
        default=[
            c
            for c in [
                "nirf_score",
                "tlr_score",
                "rpc_score",
                "go_score",
                "placement_rate_pct",
                "research_intensity_cr",
            ]
            if c in available_metrics
        ],
    )

    if len(comparison) and selected_metrics:
        long = comparison.melt(
            id_vars=["short_name"], value_vars=selected_metrics, var_name="metric", value_name="value"
        )
        st.plotly_chart(
            px.bar(long, x="short_name", y="value", color="metric", barmode="group"),
            use_container_width=True,
        )

        radar_metrics = [
            c
            for c in ["tlr_score", "rpc_score", "go_score", "oi_score", "perception_score"]
            if c in comparison.columns
        ]
        radar = go.Figure()
        for _, row in comparison.iterrows():
            radar.add_trace(
                go.Scatterpolar(
                    r=[row[c] for c in radar_metrics],
                    theta=radar_metrics,
                    fill="toself",
                    name=row["short_name"],
                )
            )
        radar.update_layout(height=470, polar=dict(radialaxis=dict(visible=True)))
        st.plotly_chart(radar, use_container_width=True)

    st.dataframe(comparison, use_container_width=True)

with tabs[1]:
    st.subheader("Exploratory visualizations")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            px.scatter(
                filtered,
                x="nirf_rank",
                y="nirf_score",
                color=color_by,
                hover_name="short_name",
                title="NIRF score vs rank",
            ),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            px.scatter(
                filtered,
                x="nirf_rank",
                y="ug_median_salary_lakh",
                color=color_by,
                size="placement_rate_pct",
                hover_name="short_name",
                title="Median salary vs rank",
            ),
            use_container_width=True,
        )

    sub_scores = [c for c in ["tlr_score", "rpc_score", "go_score", "oi_score", "perception_score"] if c in filtered]
    sub_score_long = filtered.melt(
        id_vars=["short_name", "nirf_rank"],
        value_vars=sub_scores,
        var_name="sub_score",
        value_name="score",
    )
    st.plotly_chart(
        px.line(
            sub_score_long.sort_values("nirf_rank"),
            x="nirf_rank",
            y="score",
            color="sub_score",
            markers=True,
            hover_name="short_name",
            title="NIRF sub-scores vs rank",
        ),
        use_container_width=True,
    )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            px.scatter(
                filtered,
                x="total_faculty",
                y="phd_students_fulltime",
                color=color_by,
                size="total_students",
                hover_name="short_name",
                title="Faculty strength and PhD students",
            ),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            px.scatter(
                filtered,
                x="sponsored_funding_cr",
                y="patents_granted",
                color=color_by,
                size="research_intensity_cr",
                hover_name="short_name",
                title="Research funding and patents",
            ),
            use_container_width=True,
        )

with tabs[2]:
    st.subheader("Faculty age and joining-period analysis")
    if age_df.empty:
        st.info("Faculty age-group file was not found.")
    else:
        if {"age_group", "faculty_count"}.issubset(age_df.columns):
            age_long = age_df.copy()
        else:
            id_cols = [c for c in ["short_name", "generation"] if c in age_df.columns]
            value_cols = [
                c
                for c in age_df.columns
                if c not in ["institute_id", "short_name", "city", "state", "generation"]
            ]
            age_long = age_df.melt(
                id_vars=id_cols,
                value_vars=value_cols,
                var_name="age_group",
                value_name="count",
            ).rename(columns={"count": "faculty_count"})
        st.plotly_chart(
            px.bar(
                age_long,
                x="short_name",
                y="faculty_count",
                color="age_group",
                title="Faculty age-group distribution",
            ),
            use_container_width=True,
        )
        if "generation" in age_long.columns:
            st.plotly_chart(
                px.box(
                    age_long,
                    x="generation",
                    y="faculty_count",
                    color="age_group",
                    title="Faculty age distribution by generation",
                ),
                use_container_width=True,
            )

    if join_df.empty:
        st.info("Faculty joining-period file was not found.")
    else:
        if {"joining_period", "faculty_count"}.issubset(join_df.columns):
            heatmap_data = join_df.pivot_table(
                index="short_name",
                columns="joining_period",
                values="faculty_count",
                aggfunc="sum",
                fill_value=0,
            )
        else:
            value_cols = [
                c
                for c in join_df.columns
                if c not in ["institute_id", "short_name", "city", "state", "generation"]
            ]
            heatmap_data = join_df.set_index("short_name")[value_cols]
        st.plotly_chart(
            px.imshow(
                heatmap_data,
                aspect="auto",
                color_continuous_scale="Viridis",
                title="Faculty joining-period heatmap",
            ),
            use_container_width=True,
        )

with tabs[3]:
    st.subheader("Correlation explorer")
    default_corr = [
        c
        for c in [
            "nirf_rank",
            "nirf_score",
            "tlr_score",
            "rpc_score",
            "go_score",
            "perception_score",
            "sponsored_funding_cr",
            "placement_rate_pct",
            "ug_median_salary_lakh",
        ]
        if c in all_numeric
    ]
    corr_cols = st.multiselect("Variables", all_numeric, default=default_corr)
    if len(corr_cols) >= 2:
        corr = df[corr_cols].corr(numeric_only=True)
        st.plotly_chart(
            px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1),
            use_container_width=True,
        )
        if "nirf_rank" in corr.columns:
            drivers = corr["nirf_rank"].drop("nirf_rank").abs().sort_values(ascending=False)
            st.plotly_chart(
                px.bar(
                    drivers.reset_index(),
                    x="index",
                    y="nirf_rank",
                    title="Rank drivers by absolute correlation",
                    labels={"index": "variable", "nirf_rank": "absolute correlation"},
                ),
                use_container_width=True,
            )
        x_axis = st.selectbox("X variable", corr_cols)
        y_axis = st.selectbox("Y variable", corr_cols, index=min(1, len(corr_cols) - 1))
        st.plotly_chart(
            px.scatter(df, x=x_axis, y=y_axis, color="generation", hover_name="short_name"),
            use_container_width=True,
        )
    else:
        st.info("Select at least two numeric variables.")

with tabs[4]:
    st.subheader("PCA and KMeans clustering")
    feature_options = [c for c in CORE_FEATURES if c in df.columns]
    selected_features = st.multiselect(
        "Features for PCA/KMeans", feature_options, default=feature_options[:12]
    )
    k = st.slider("Number of clusters", 2, 6, 3)

    if len(selected_features) >= 2:
        clustered, pca_df, scaled, _ = clustered_data(df, k, selected_features)
        _, loadings, variance, _, _ = pca_projection(df, selected_features)
        pca_df["generation"] = df["generation"]

        st.write(f"Explained variance: PC1 {variance[0] * 100:.1f}%, PC2 {variance[1] * 100:.1f}%")
        st.plotly_chart(
            px.scatter(
                pca_df,
                x="PC1",
                y="PC2",
                color="cluster_label",
                symbol="generation",
                hover_name="short_name",
                title="2D PCA scatter with KMeans clusters",
            ),
            use_container_width=True,
        )

        loading_plot = (
            loadings.assign(strength=loadings["PC1"].abs() + loadings["PC2"].abs())
            .sort_values("strength", ascending=False)
            .head(12)
            .reset_index()
        )
        st.plotly_chart(
            px.bar(loading_plot, x="index", y=["PC1", "PC2"], barmode="group", title="PCA loadings"),
            use_container_width=True,
        )

        silhouette_rows = []
        for candidate_k in range(2, min(7, len(df))):
            labels = KMeans(n_clusters=candidate_k, random_state=42, n_init=20).fit_predict(scaled)
            silhouette_rows.append(
                {"k": candidate_k, "silhouette_score": silhouette_score(scaled, labels)}
            )
        st.plotly_chart(
            px.line(pd.DataFrame(silhouette_rows), x="k", y="silhouette_score", markers=True),
            use_container_width=True,
        )

        profile_cols = [
            c
            for c in [
                "nirf_score",
                "tlr_score",
                "rpc_score",
                "go_score",
                "placement_rate_pct",
                "research_intensity_cr",
                "ug_median_salary_lakh",
            ]
            if c in clustered.columns
        ]
        st.plotly_chart(
            px.imshow(
                clustered.groupby("cluster_label")[profile_cols].mean(),
                aspect="auto",
                color_continuous_scale="YlGnBu",
                title="Cluster feature heatmap",
            ),
            use_container_width=True,
        )
        st.dataframe(
            clustered[["short_name", "nirf_rank", "generation", "cluster_label"] + profile_cols]
            .sort_values(["cluster_label", "nirf_rank"]),
            use_container_width=True,
        )
    else:
        st.info("Select at least two features.")

with tabs[5]:
    st.subheader("Ranking sensitivity simulator")
    sim_cols = [
        c
        for c in [
            "tlr_score",
            "rpc_score",
            "go_score",
            "oi_score",
            "perception_score",
            "placement_rate_pct",
            "ug_median_salary_lakh",
            "research_intensity_cr",
            "phd_output_rate_pct",
            "patents_granted",
        ]
        if c in df.columns
    ]

    sliders = st.columns(2)
    weights = {}
    for index, col in enumerate(sim_cols):
        default = 20 if col in {"tlr_score", "rpc_score", "go_score"} else 10
        weights[col] = sliders[index % 2].slider(col, 0, 100, default)

    total_weight = sum(weights.values())
    if total_weight == 0:
        st.warning("Set at least one weight above zero.")
    else:
        norm = normalized_scores(df, sim_cols)
        simulated_score = sum(norm[col] * (weights[col] / total_weight) for col in sim_cols)
        simulated = df[["short_name", "nirf_rank", "nirf_score", "generation"]].copy()
        simulated["simulated_score"] = simulated_score * 100
        simulated["simulated_rank"] = (
            simulated["simulated_score"].rank(ascending=False, method="min").astype(int)
        )
        simulated["rank_change"] = simulated["nirf_rank"] - simulated["simulated_rank"]
        simulated = simulated.sort_values("simulated_rank")

        st.plotly_chart(
            px.bar(
                simulated.head(15),
                x="short_name",
                y="simulated_score",
                color="rank_change",
                title="Top simulated rankings",
            ),
            use_container_width=True,
        )
        st.dataframe(simulated, use_container_width=True)
        st.download_button(
            "Download simulated rankings",
            simulated.to_csv(index=False),
            "simulated_iit_rankings.csv",
            "text/csv",
        )

with tabs[6]:
    st.subheader("Missing-value audit and research insights")
    missing = df.isna().sum().reset_index()
    missing.columns = ["column", "missing_values"]
    missing = missing[missing["missing_values"] > 0].sort_values("missing_values", ascending=False)
    if len(missing):
        st.dataframe(missing, use_container_width=True)
    else:
        st.success("No missing values found in the master dataset.")

    clustered, _, _, _ = clustered_data(df, 3)
    profile_cols = [
        c
        for c in [
            "nirf_rank",
            "nirf_score",
            "tlr_score",
            "rpc_score",
            "go_score",
            "placement_rate_pct",
            "research_intensity_cr",
            "ug_median_salary_lakh",
        ]
        if c in clustered.columns
    ]
    st.write("Cluster profile averages")
    st.dataframe(clustered.groupby("cluster_label")[profile_cols].mean().round(2), use_container_width=True)

    best_score = df.sort_values("nirf_score", ascending=False).iloc[0]
    best_salary = df.sort_values("ug_median_salary_lakh", ascending=False).iloc[0]
    best_research = df.sort_values("research_intensity_cr", ascending=False).iloc[0]
    st.markdown(
        f"""
- Highest NIRF score: **{best_score["short_name"]}** ({best_score["nirf_score"]:.2f}).
- Highest UG median salary: **{best_salary["short_name"]}** ({best_salary["ug_median_salary_lakh"]:.2f} lakh).
- Highest research intensity: **{best_research["short_name"]}** ({best_research["research_intensity_cr"]:.2f} crore).
- The ranking simulator shows how ranks shift when teaching, research, outcomes, perception, placement, and patent indicators receive different weights.
"""
    )

with tabs[7]:
    st.subheader("Data table and export")
    st.dataframe(filtered.sort_values("nirf_rank"), use_container_width=True)
    st.download_button(
        "Download filtered data",
        filtered.to_csv(index=False),
        "filtered_iit_nirf_2025.csv",
        "text/csv",
    )
