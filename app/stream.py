# =========================================================
# IMPORT LIBRARIES
# =========================================================
import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..'
        )
    )
)
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.model import predict_dataset

from src.similarity import (
    clean_similarity_data,
    create_similarity_vectors,
    compute_similarity_matrix,
    find_similar_reviews
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Review Analytics System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS — BOLD & COLORFUL THEME
# =========================================================

st.markdown("""
<style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500;600&display=swap');

    /* ── Root palette ── */
    :root {
        --ink:       #0D0D12;
        --surface:   #F7F6FF;
        --card:      #FFFFFF;
        --violet:    #6B4FFF;
        --coral:     #FF5C5C;
        --teal:      #00C9A7;
        --amber:     #FFB800;
        --border:    #E8E5FF;
        --muted:     #8B8AA3;
    }

    /* ── Base ── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: var(--surface);
        color: var(--ink);
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container {
        padding: 2rem 3rem 4rem 3rem;
        max-width: 1300px;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: var(--ink);
        border-right: none;
    }
    [data-testid="stSidebar"] * {
        color: #E0DEFF !important;
        font-family: 'DM Sans', sans-serif;
    }
    [data-testid="stSidebar"] .sidebar-brand {
        font-family: 'Syne', sans-serif;
        font-size: 1.3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #fff !important;
        padding: 1.5rem 1.5rem 0.5rem;
        display: block;
    }
    [data-testid="stSidebar"] .sidebar-tagline {
        font-size: 0.75rem;
        color: var(--muted) !important;
        padding: 0 1.5rem 2rem;
        display: block;
        line-height: 1.5;
    }
    [data-testid="stSidebar"] hr {
        border-color: #1E1E2E;
        margin: 0.5rem 1.5rem;
    }
    [data-testid="stSidebar"] .sidebar-label {
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #4B4A6A !important;
        padding: 1.5rem 1.5rem 0.5rem;
        display: block;
    }
    [data-testid="stSidebar"] .sidebar-step {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.5rem 1.5rem;
        font-size: 0.85rem;
        color: #A0A0C0 !important;
    }
    [data-testid="stSidebar"] .sidebar-step .dot {
        width: 6px; height: 6px;
        border-radius: 50%;
        background: #6B4FFF;
        flex-shrink: 0;
    }

    /* ── Hero heading ── */
    .hero-heading {
        font-family: 'Syne', sans-serif;
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        line-height: 1.1;
        color: var(--ink);
        margin-bottom: 0.4rem;
    }
    .hero-heading span {
        background: linear-gradient(90deg, #6B4FFF, #FF5C5C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-sub {
        font-size: 0.95rem;
        color: var(--muted);
        margin-bottom: 2rem;
        max-width: 560px;
        line-height: 1.6;
    }

    /* ── Pill badges ── */
    .pill-row {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-bottom: 2.5rem;
    }
    .pill {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        padding: 0.3rem 0.75rem;
        border-radius: 999px;
        border: 1.5px solid;
    }
    .pill-v { color: #6B4FFF; border-color: #6B4FFF; background: #F0EDFF; }
    .pill-t { color: #00976C; border-color: #00C9A7; background: #E6FFF9; }
    .pill-a { color: #B37A00; border-color: #FFB800; background: #FFF8E0; }
    .pill-c { color: #C43E3E; border-color: #FF5C5C; background: #FFF0F0; }

    /* ── Section header ── */
    .section-header {
        font-family: 'Syne', sans-serif;
        font-size: 1.25rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: var(--ink);
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }

    /* ── Metric cards ── */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: var(--card);
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        border: 1.5px solid var(--border);
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 4px;
        border-radius: 16px 16px 0 0;
    }
    .mc-v::before { background: #6B4FFF; }
    .mc-c::before { background: #FF5C5C; }
    .mc-g::before { background: #00C9A7; }
    .mc-a::before { background: #FFB800; }
    .metric-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-family: 'Syne', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1;
    }
    .mv-v { color: #6B4FFF; }
    .mv-c { color: #FF5C5C; }
    .mv-g { color: #00976C; }
    .mv-a { color: #B37A00; }
    .metric-icon {
        position: absolute;
        right: 1rem;
        top: 1.2rem;
        font-size: 1.8rem;
        opacity: 0.12;
    }

    /* ── Upload area ── */
    [data-testid="stFileUploadDropzone"] {
        background: var(--card) !important;
        border: 2px dashed #C4BBFF !important;
        border-radius: 16px !important;
        transition: border-color 0.2s;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #6B4FFF !important;
    }

    /* ── Success / info boxes ── */
    [data-testid="stAlert"] {
        border-radius: 12px !important;
        border-left: 4px solid #6B4FFF !important;
    }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1.5px solid var(--border);
    }

    /* ── Subheader override ── */
    h2, h3 {
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }

    /* ── Chart container card ── */
    .chart-card {
        background: #FFFFFF;
    border-radius: 20px;
    padding: 10px 14px 14px 14px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.08);
    }

    /* ── Similarity badge ── */
    .sim-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: #FFF0F0;
        color: #C43E3E;
        border: 1.5px solid #FF5C5C;
        border-radius: 999px;
        padding: 0.4rem 1rem;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .sim-badge .dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        background: #FF5C5C;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.3); }
    }

    /* ── Progress bar ── */
    .prog-wrap {
        background: #F0EDFF;
        border-radius: 999px;
        height: 10px;
        width: 100%;
        margin-top: 0.6rem;
        overflow: hidden;
    }
    .prog-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #6B4FFF, #FF5C5C);
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        color: var(--muted);
        font-size: 0.75rem;
        margin-top: 4rem;
        padding-top: 1.5rem;
        border-top: 1.5px solid var(--border);
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown('<span class="sidebar-brand">🔍 ReviewRadar</span>', unsafe_allow_html=True)
    st.markdown(
        '<span class="sidebar-tagline">Suspicious review detection powered by NLP & Linear SVM</span>',
        unsafe_allow_html=True
    )
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<span class="sidebar-label">Pipeline Steps</span>', unsafe_allow_html=True)

    steps = [
        "Upload CSV dataset",
        "NLP preprocessing",
        "Feature engineering",
        "SVM classification",
        "Cosine similarity scan",
        "Review results",
    ]
    for step in steps:
        st.markdown(
            f'<div class="sidebar-step"><div class="dot"></div>{step}</div>',
            unsafe_allow_html=True
        )

    st.markdown("<hr style='margin-top:2rem'>", unsafe_allow_html=True)
    st.markdown('<span class="sidebar-label">About</span>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-step">Detects fake, promotional, and duplicate reviews using ML signals.</div>',
        unsafe_allow_html=True
    )


# =========================================================
# HERO HEADER
# =========================================================

st.markdown("""
<div class="hero-heading">Review <span>Analytics</span><br>& Fraud Detection</div>
<div class="hero-sub">
    Upload your product review dataset and let the pipeline identify suspicious,
    promotional, and duplicate reviews — instantly.
</div>
<div class="pill-row">
    <span class="pill pill-v">NLP Preprocessing</span>
    <span class="pill pill-t">Feature Engineering</span>
    <span class="pill pill-a">Cosine Similarity</span>
    <span class="pill pill-c">Linear SVM</span>
</div>
""", unsafe_allow_html=True)


# =========================================================
# FILE UPLOADER
# =========================================================

uploaded_file = st.file_uploader(
    "Drop your review dataset here (.csv)",
    type=["csv"],
    help="Upload a CSV file containing product reviews with a 'reviewText' column."
)


# =========================================================
# MAIN PIPELINE
# =========================================================

if uploaded_file is not None:

    # ── Load ──
    df = pd.read_csv(uploaded_file)
    st.success(f"✅  Dataset loaded — **{len(df):,}** rows detected.")

    # ── Preview ──
    # st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-header"><div class="accent-bar bar-v"></div>Dataset Preview</div>',
        unsafe_allow_html=True
    )
    st.subheader("📊 Visual Analytics")

    st.caption(
    "Interactive visual analysis of review ratings and suspicious activity distribution."
    )
    st.subheader("📂 Dataset Preview")

#     st.caption(
#     "Preview uploaded reviews and metadata before running suspicious review analysis."
# )

    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(df.head(8), use_container_width=True)

    # ── Run pipeline ──
    # st.markdown('<hr class="section-divider">', unsafe_allow_html=True)  #need to comeback
    st.markdown(
        '<div class="section-header"><div class="accent-bar bar-a"></div>Running Prediction Pipeline</div>',
        unsafe_allow_html=True
    )

    with st.spinner("Classifying reviews with Linear SVM…"):
        predicted_df = predict_dataset(df)

    st.success("🎯  Prediction pipeline complete.")

    # ── Compute metrics ──
    total_reviews      = len(predicted_df)
    spam_count = (
    predicted_df['prediction'] == 1).sum()
    non_spam_count = (
    predicted_df['prediction'] == 0
    ).sum()
    suspicious_pct     = (spam_count / total_reviews) * 100

    # ── Metric cards ──
    # st.markdown('<hr class="section-divider">', unsafe_allow_html=True) same here
    # st.markdown(
    #     '<div class="section-header"><div class="accent-bar bar-v"></div>Summary Metrics</div>',
    #     unsafe_allow_html=True
    # )
    st.subheader("📊 Summary Metrics")

    st.caption(
    "Overview of suspicious review activity detected in the uploaded dataset."
    )

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card mc-v">
            <div class="metric-label">Total Reviews</div>
            <div class="metric-value mv-v">{total_reviews:,}</div>
            <div class="metric-icon">📋</div>
        </div>
        <div class="metric-card mc-c">
            <div class="metric-label">Spam</div>
            <div class="metric-value mv-c">{spam_count:,}</div>
            <div class="metric-icon">🚨</div>
            <div class="prog-wrap">
                <div class="prog-fill" style="width:{suspicious_pct:.1f}%"></div>
            </div>
        </div>
        <div class="metric-card mc-g">
            <div class="metric-label">Non-Spam</div>
            <div class="metric-value mv-g">{non_spam_count:,}</div>
            <div class="metric-icon">✅</div>
        </div>
        <div class="metric-card mc-a">
            <div class="metric-label">Spam Rate</div>
            <div class="metric-value mv-a">{suspicious_pct:.1f}%</div>
            <div class="metric-icon">📊</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


    # =====================================================
    # SUSPICIOUS REVIEW MODERATION TABLE
    # =====================================================

    st.subheader("🚨 Predicted Spam Review Moderation Table")

    st.caption(
    "Reviews flagged for potential suspicious behavior requiring inspection."
    )

    moderation_df = predicted_df[
        predicted_df['prediction'] == 1
    ].copy()

    moderation_df = moderation_df[
        [
            'reviewerID',
            'asin',
            'reviewText',
            'prediction_confidence',
            'reviewer_trust_score',
            'suspicious_reasons'
        ]
    ]

    moderation_df.columns = [
        'Reviewer',
        'Product',
        'Review',
        'ML Confidence',
        'Trust Score',
        'Reasons Flagged'
    ]

    st.dataframe(
        moderation_df.head(50),
        use_container_width=True,
        height=400
    )
    # ── Charts ──
    # st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-header"><div class="accent-bar bar-t"></div>Visual Analytics</div>',
        unsafe_allow_html=True
    )

    col_a, col_b = st.columns(2)
    
    # Chart palette
    VIOLET = "#6B4FFF"
    CORAL  = "#FF5C5C"
    TEAL   = "#00C9A7"
    AMBER  = "#FFB800"
    BG     = "#FFFFFF"
    INK    = "#0D0D12"
    
    # =====================================================
    # ADJUSTED PRODUCT RATINGS
    # =====================================================

    st.subheader("⭐ Adjusted Product Ratings")

    st.caption(
    "Estimated product ratings after excluding suspicious reviews."
    )

    product_analysis = []

    products = predicted_df['asin'].unique()

    for product in products:

        product_df = predicted_df[
            predicted_df['asin'] == product
        ]

        original_rating = (
            product_df['overall']
            .mean()
        )

        clean_df = product_df[
            product_df['prediction'] == 0
        ]

        if len(clean_df) > 0:

            adjusted_rating = (
                clean_df['overall']
                .mean()
            )

        else:

            adjusted_rating = 0

        suspicious_reviews = (
            product_df['prediction'] == 0
        ).sum()

        product_analysis.append({

            'Product': product,
            'Original Rating': round(original_rating, 2),
            'Adjusted Rating': round(adjusted_rating, 2),
            'Suspicious Reviews': suspicious_reviews
        })

    product_analysis_df = pd.DataFrame(
        product_analysis
    )

    st.dataframe(
        product_analysis_df.sort_values(
            by='Suspicious Reviews',
            ascending=False
        ).head(20),
        use_container_width=True
)
    # st.subheader("📊 Visual Analytics")

    # st.caption(
    # "Interactive visual analysis of review ratings and suspicious activity distribution."
    # )

    st.markdown("<br>", unsafe_allow_html=True)
    # Rating distribution
    st.subheader("📊 Visual Analytics")

    st.caption(
    "Interactive visual analysis of review ratings and suspicious activity distribution."
    )
    with col_a:
        st.markdown(
        "### ⭐ Rating Distribution"
        )

        st.caption(
        "Distribution of uploaded reviews across star ratings."
        )
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)

        counts  = predicted_df['overall'].value_counts().sort_index()
        bars    = ax.bar(counts.index, counts.values,
                         color=[VIOLET, TEAL, AMBER, CORAL, "#B44FFF"],
                         width=0.6, zorder=3)

        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + counts.max() * 0.01,
                    f'{int(bar.get_height()):,}',
                    ha='center', va='bottom',
                    fontsize=9, color=INK, fontweight='600')

        ax.set_xlabel("Star Rating", fontsize=9, color="#8B8AA3", labelpad=6)
        ax.set_ylabel("Count", fontsize=9, color="#8B8AA3", labelpad=6)
        # ax.set_title("Rating Distribution", fontsize=12, fontweight='700',
        #              color=INK, pad=12)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.spines['bottom'].set_color("#E8E5FF")
        ax.tick_params(colors="#8B8AA3", labelsize=9)
        ax.yaxis.grid(True, color="#F0EDFF", zorder=0)
        plt.tight_layout(pad=1)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    # Prediction distribution — donut
    with col_b:
        st.markdown(
        "### 🥧 Prediction Breakdown"
        )

        st.caption(
        "Comparison between suspicious and genuine review predictions."
        )
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(5.5, 3.5))
        fig2.patch.set_facecolor(BG)
        ax2.set_facecolor(BG)

        sizes = [
                spam_count,
                non_spam_count
        ]
        colors = [CORAL, TEAL]
        wedges, texts, autotexts = ax2.pie(
            sizes,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            pctdistance=0.78,
            wedgeprops=dict(width=0.55, edgecolor=BG, linewidth=3)
        )
        for at in autotexts:
            at.set_fontsize(10)
            at.set_fontweight('700')
            at.set_color(BG)

        legend_patches = [
            mpatches.Patch(color=CORAL, label=f'Spam  {spam_count:,}'),
            mpatches.Patch(color=TEAL,  label=f'Non-Spam  {non_spam_count:,}'),
        ]
        ax2.legend(handles=legend_patches, loc='lower center',
                   ncol=2, fontsize=8.5, frameon=False,
                   bbox_to_anchor=(0.5, -0.08))
        # ax2.set_title("Prediction Breakdown", fontsize=12, fontweight='700',
        #               color=INK, pad=12)
        plt.tight_layout(pad=1)
        st.pyplot(fig2)
        st.markdown('</div>', unsafe_allow_html=True)

    # =====================================================
    # REVIEWER TRUST ANALYTICS
    # =====================================================

    st.subheader("👤 Reviewer Trust Analytics")

    st.caption(
    "Reviewer-level behavioral analysis based on trust indicators and review activity."
    )

    reviewer_analysis = predicted_df.groupby(
        'reviewerID'
    ).agg({

        'reviewer_trust_score': 'mean',
        'reviewText': 'count'

    }).reset_index()

    reviewer_analysis.columns = [
        'Reviewer',
        'Average Trust Score',
        'Review Count'
    ]

    st.dataframe(

        reviewer_analysis.sort_values(
            by='Average Trust Score'
        ).head(20),

        use_container_width=True
    )

    # ── Similarity detection ──
    st.subheader("🔗 Coordinated Review Pattern Analysis")

    st.caption(
    "Detection of unusually similar review language that may indicate coordinated campaigns."
    )

    with st.spinner("Computing cosine similarity matrix…"):
        similarity_df     = clean_similarity_data(predicted_df.copy())
        sample_size       = min(1000, len(similarity_df))
        sample_df         = similarity_df.sample(
                                n=sample_size, random_state=42
                            ).reset_index(drop=True)
        X_similarity      = create_similarity_vectors(sample_df['processed_review'])
        similarity_matrix = compute_similarity_matrix(X_similarity)
        similarity_results = find_similar_reviews(
                                sample_df, similarity_matrix, threshold=0.65
                             )

    pair_count = len(similarity_results)

    st.markdown(f"""
    <div class="sim-badge">
        <div class="dot"></div>
        {pair_count:,} potential coordinated review patterns detected
    </div>
    """, unsafe_allow_html=True)

    if pair_count > 0:
        display_similarity = similarity_results[
        [
            'reviewer_1',
            'product_1',
            'similarity_score',
            'review_1',
            'review_2'
        ]
    ]

        display_similarity.columns = [
            'Reviewer',
            'Product',
            'Similarity Score',
            'Review A',
            'Review B'
       ]

        st.dataframe(
            display_similarity.head(20),
            use_container_width=True,
            height=340
        )

    else:
        st.info("No highly similar reviews found")


    # ── Footer ──
    st.markdown(
        '<div class="footer">ReviewRadar · NLP + SVM Pipeline · Built with Streamlit</div>',
        unsafe_allow_html=True
    )

else:
    # Empty state illustration
    st.markdown("""
    <div style="
        text-align: center;
        padding: 4rem 2rem;
        background: #FFFFFF;
        border-radius: 20px;
        border: 2px dashed #C4BBFF;
        margin-top: 1rem;
    ">
        <div style="font-size: 3rem; margin-bottom: 1rem;">📂</div>
        <div style="font-family: 'Syne', sans-serif; font-size: 1.2rem;
                    font-weight: 700; color: #0D0D12; margin-bottom: 0.5rem;">
            No dataset loaded yet
        </div>
        <div style="color: #8B8AA3; font-size: 0.9rem; max-width: 360px; margin: 0 auto;">
            Upload a CSV file above to kick off the full analytics pipeline.
            Results will appear here instantly.
        </div>
    </div>
    """, unsafe_allow_html=True)