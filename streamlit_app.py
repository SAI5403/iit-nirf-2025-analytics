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


def find_first_data_file(filenames: list[str]) -> Path | None:
    for filename in filenames:
        path = find_data_file(filename)
        if path.exists():
            return path
    return None


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


@st.cache_data
def load_twitter_data():
    path = find_first_data_file(
        [
            "iit_x_followers_2026-07-09.csv",
            "iit_x_followers.csv",
            "iit_twitter_followers.csv",
            "twitter_analysis.csv",
        ]
    )
    if path is None:
        return pd.DataFrame()
    out = pd.read_csv(path)
    for col in ["followers_count", "posts_count"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


@st.cache_data
def load_google_trends_data():
    files_map = {
        "group1": "google_trends_group1.csv",
        "group2": "google_trends_group2.csv",
        "group3": "google_trends_group3.csv",
        "group4": "google_trends_group4.csv",
        "group5": "google_trends_group5.csv",
        "group6": "google_trends_group6.csv",
    }
    paths = {group: find_data_file(filename) for group, filename in files_map.items()}
    if not all(path.exists() for path in paths.values()):
        monthly_path = find_first_data_file(["iit_google_trends_monthly_normalized.csv"])
        if monthly_path is None:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        monthly = pd.read_csv(monthly_path)
        time_col = "Time" if "Time" in monthly.columns else monthly.columns[0]
        monthly[time_col] = pd.to_datetime(monthly[time_col])
        monthly = monthly.set_index(time_col)
    else:
        rename_map = {
            "Indian Institute Of Technology Delhi (IIT Delhi)": "IIT Delhi",
            "Indian Institute of Technology Kanpur": "IIT Kanpur",
            "Indian Institute of Technology, Kharagpur": "IIT Kharagpur",
            "Indian Institute Of Technologyâ€“Madras (IITâ€“Madras)": "IIT Madras",
            "Indian Institute of Technology Bombay": "IIT Bombay",
            "Indian Institute of Technology Guwahati": "IIT Guwahati",
            "Indian Institute Of Technology Roorkee": "IIT Roorkee",
            "Indian Institute Of Technologyâ€“Ropar (IITâ€“Ropar)": "IIT Ropar",
            "Indian Institute of Technology, Patna": "IIT Patna",
            "Indian Institute Of Technology Gandhinagar (IIT Gandhinagar)": "IIT Gandhinagar",
            "Indian Institute of Technology, Hyderabad": "IIT Hyderabad",
            "Indian Institute of Technology (IIT), Jodhpur": "IIT Jodhpur",
            "Indian Institute Of Technology (IIT) Bhubaneswar": "IIT Bhubaneswar",
            "Indian Institute of Technology Indore": "IIT Indore",
            "Indian Institute of Technology Mandi": "IIT Mandi",
            "Indian Institute of Technology (BHU) Varanasi": "IIT BHU",
            "Indian Institute Of Technologyâ€“Palakkad (IITâ€“Palakkad)": "IIT Palakkad",
            "Indian Institute Of Technology (IIT) Tirupati": "IIT Tirupati",
            "Indian Institute of Technology (Indian School of Mines), Dhanbad": "IIT Dhanbad",
            "Indian Institute of Technology Bhilai": "IIT Bhilai",
            "Indian Institute Of Technology Goa": "IIT Goa",
            "Indian Institute of Technology, Jammu": "IIT Jammu",
            "Indian Institute Of Technology Dharwad": "IIT Dharwad",
        }
        iit_order = [
            "IIT Kharagpur",
            "IIT Bombay",
            "IIT Madras",
            "IIT Kanpur",
            "IIT Delhi",
            "IIT Guwahati",
            "IIT Roorkee",
            "IIT Ropar",
            "IIT Bhubaneswar",
            "IIT Gandhinagar",
            "IIT Hyderabad",
            "IIT Jodhpur",
            "IIT Patna",
            "IIT Indore",
            "IIT Mandi",
            "IIT BHU",
            "IIT Palakkad",
            "IIT Tirupati",
            "IIT Dhanbad",
            "IIT Bhilai",
            "IIT Goa",
            "IIT Jammu",
            "IIT Dharwad",
        ]
        group_dfs = {}
        for group, path in paths.items():
            trends = pd.read_csv(path)
            trends["Time"] = pd.to_datetime(trends["Time"])
            trends = trends.set_index("Time").rename(columns=rename_map)
            for col in trends.columns:
                trends[col] = pd.to_numeric(trends[col], errors="coerce")
            group_dfs[group] = trends

        anchor_name = "IIT Delhi"
        monthly = group_dfs["group1"].copy()
        master_anchor = monthly[anchor_name]
        for group_name, trends in group_dfs.items():
            if group_name == "group1":
                continue
            common_dates = master_anchor.index.intersection(trends.index)
            anchor_master = master_anchor.loc[common_dates]
            anchor_group = trends.loc[common_dates, anchor_name]
            valid = (anchor_master > 0) & (anchor_group > 0)
            scale_factor = (anchor_master[valid] / anchor_group[valid]).median() if valid.sum() else 1
            scaled = trends * scale_factor
            for col in scaled.columns:
                if col != anchor_name:
                    monthly[col] = scaled[col]
        monthly = monthly[[col for col in iit_order if col in monthly.columns]]

    monthly.index.name = "Time"
    avg = monthly.mean().sort_values(ascending=False).reset_index()
    avg.columns = ["iit", "average_google_trends_interest"]
    overall = monthly.mean(axis=1).reset_index()
    overall.columns = ["month", "average_google_trends_interest"]
    peak = monthly.idxmax().reset_index()
    peak.columns = ["iit", "peak_month"]
    peak["peak_value"] = [monthly.loc[row["peak_month"], row["iit"]] for _, row in peak.iterrows()]
    peak["peak_month"] = pd.to_datetime(peak["peak_month"]).dt.strftime("%B %Y")

    season_source = monthly.copy()
    season_source["month"] = season_source.index.month_name()
    month_order = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    seasonality = (
        season_source.groupby("month")[monthly.columns]
        .mean()
        .mean(axis=1)
        .reindex(month_order)
        .reset_index()
    )
    seasonality.columns = ["month", "average_google_trends_interest"]
    return monthly, avg, overall, seasonality


@st.cache_data
def load_optional_table(filenames: list[str]):
    path = find_first_data_file(filenames)
    if path is None:
        return pd.DataFrame()
    return pd.read_csv(path)


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
        "Search & Social",
        "Campus & Research",
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
    st.subheader("Google Trends and X/Twitter analysis")

    x_df = load_twitter_data()
    monthly_trends, avg_trends, overall_trends, seasonality = load_google_trends_data()

    st.write("X/Twitter official account metrics")
    if x_df.empty:
        st.info(
            "Twitter/X follower data was not found. Upload one of these files to data/: "
            "iit_x_followers_2026-07-09.csv, iit_x_followers.csv, or iit_twitter_followers.csv."
        )
    else:
        x_cols = st.columns(3)
        x_cols[0].metric("Accounts tracked", len(x_df))
        if "followers_count" in x_df.columns:
            top_followers = x_df.sort_values("followers_count", ascending=False).iloc[0]
            x_cols[1].metric("Most followed", top_followers["short_name"])
        if "posts_count" in x_df.columns:
            top_posts = x_df.sort_values("posts_count", ascending=False).iloc[0]
            x_cols[2].metric("Most active", top_posts["short_name"])

        if {"short_name", "followers_count"}.issubset(x_df.columns):
            st.plotly_chart(
                px.bar(
                    x_df.sort_values("followers_count", ascending=True),
                    x="followers_count",
                    y="short_name",
                    orientation="h",
                    title="Official X/Twitter followers across IITs",
                ),
                use_container_width=True,
            )
        if {"short_name", "posts_count"}.issubset(x_df.columns):
            st.plotly_chart(
                px.bar(
                    x_df.sort_values("posts_count", ascending=True),
                    x="posts_count",
                    y="short_name",
                    orientation="h",
                    title="Official X/Twitter posting activity",
                ),
                use_container_width=True,
            )
        if {"posts_count", "followers_count", "short_name"}.issubset(x_df.columns):
            st.plotly_chart(
                px.scatter(
                    x_df,
                    x="posts_count",
                    y="followers_count",
                    hover_name="short_name",
                    text="short_name",
                    title="Posting activity vs followers",
                ),
                use_container_width=True,
            )
        st.dataframe(x_df, use_container_width=True)

    st.write("Google Trends search-interest analysis")
    if monthly_trends.empty:
        st.info(
            "Google Trends files were not found. Upload google_trends_group1.csv through "
            "google_trends_group6.csv to data/."
        )
    else:
        selected_trend_iits = st.multiselect(
            "Select IITs for trend line",
            monthly_trends.columns.tolist(),
            default=avg_trends["iit"].head(5).tolist(),
        )
        if selected_trend_iits:
            trend_long = (
                monthly_trends[selected_trend_iits]
                .reset_index()
                .melt(id_vars=monthly_trends.index.name or "Time", var_name="iit", value_name="interest")
            )
            st.plotly_chart(
                px.line(
                    trend_long,
                    x=monthly_trends.index.name or "Time",
                    y="interest",
                    color="iit",
                    title="Monthly Google Trends interest",
                ),
                use_container_width=True,
            )
        st.plotly_chart(
            px.bar(
                avg_trends.sort_values("average_google_trends_interest", ascending=True),
                x="average_google_trends_interest",
                y="iit",
                orientation="h",
                title="Average Google search interest",
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            px.line(
                overall_trends,
                x="month",
                y="average_google_trends_interest",
                markers=True,
                title="Overall IIT search-interest trend",
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            px.bar(
                seasonality,
                x="month",
                y="average_google_trends_interest",
                title="Month-of-year search seasonality",
            ),
            use_container_width=True,
        )

with tabs[7]:
    st.subheader("Campus buildings, accessibility, and research metrics")

    building_df = load_optional_table(
        [
            "iit_building_count_results_2023.csv",
            "iit_building_counts_2023.csv",
            "iit_buildings_count_2023.csv",
            "iit_open_buildings_2023.csv",
        ]
    )
    area_df = load_optional_table(["iit_reported_campus_area_wikipedia.csv"])
    airport_df = load_optional_table(["iit_nearest_airports_updated.csv", "iit_nearest_airports.csv"])
    openalex_df = load_optional_table(
        ["iit_openalex_wikipedia_metrics_updated.csv", "iit_openalex_metrics.csv"]
    )

    if building_df.empty:
        st.info(
            "Building-count results were not found. Upload a generated CSV such as "
            "iit_building_count_results_2023.csv to data/. The dashboard does not run "
            "Earth Engine during deployment."
        )
    else:
        count_col = (
            "estimated_building_count"
            if "estimated_building_count" in building_df.columns
            else "building_count"
        )
        if count_col in building_df.columns and "short_name" in building_df.columns:
            building_df["building_count_rounded"] = pd.to_numeric(
                building_df[count_col], errors="coerce"
            ).round()
            st.plotly_chart(
                px.bar(
                    building_df.sort_values("building_count_rounded", ascending=True),
                    x="building_count_rounded",
                    y="short_name",
                    orientation="h",
                    title="Approximate building counts across IIT campuses",
                ),
                use_container_width=True,
            )
            if not area_df.empty and "reported_area_acres" in area_df.columns:
                merged_building_area = building_df.merge(
                    area_df[["short_name", "reported_area_acres"]],
                    on="short_name",
                    how="left",
                )
                st.plotly_chart(
                    px.scatter(
                        merged_building_area,
                        x="reported_area_acres",
                        y="building_count_rounded",
                        hover_name="short_name",
                        title="Building count vs reported campus area",
                    ),
                    use_container_width=True,
                )
        st.dataframe(building_df, use_container_width=True)

    if airport_df.empty:
        st.info("Nearest-airport data was not found. Upload iit_nearest_airports_updated.csv to data/.")
    else:
        if {"short_name", "distance_to_airport_km"}.issubset(airport_df.columns):
            st.plotly_chart(
                px.bar(
                    airport_df.sort_values("distance_to_airport_km", ascending=False),
                    x="short_name",
                    y="distance_to_airport_km",
                    color="nearest_airport_iata" if "nearest_airport_iata" in airport_df.columns else None,
                    title="Distance from IIT campus to nearest major airport",
                ),
                use_container_width=True,
            )
        st.dataframe(airport_df, use_container_width=True)

    if openalex_df.empty:
        st.info(
            "OpenAlex/Wikipedia research metrics were not found. Upload "
            "iit_openalex_wikipedia_metrics_updated.csv to data/."
        )
    else:
        metric_candidates = [
            c
            for c in ["works_count", "cited_by_count", "h_index", "i10_index", "wikipedia_pageviews"]
            if c in openalex_df.columns
        ]
        selected_metric = st.selectbox("Research metric", metric_candidates) if metric_candidates else None
        if selected_metric and "short_name" in openalex_df.columns:
            st.plotly_chart(
                px.bar(
                    openalex_df.sort_values(selected_metric, ascending=True),
                    x=selected_metric,
                    y="short_name",
                    orientation="h",
                    title=f"{selected_metric} across IITs",
                ),
                use_container_width=True,
            )
        st.dataframe(openalex_df, use_container_width=True)

with tabs[8]:
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

with tabs[9]:
    st.subheader("Data table and export")
    st.dataframe(filtered.sort_values("nirf_rank"), use_container_width=True)
    st.download_button(
        "Download filtered data",
        filtered.to_csv(index=False),
        "filtered_iit_nirf_2025.csv",
        "text/csv",
    )
