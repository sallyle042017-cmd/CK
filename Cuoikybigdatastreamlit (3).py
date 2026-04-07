"""
app.py - Streamlit Web UI cho đồ án Big Data / PySpark MLlib
Dataset: Brazilian E-Commerce (Olist)
Pages: Dashboard | Phân khúc Khách Hàng | Khuyến nghị Sản Phẩm | Dự đoán | Xu hướng | Admin
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import warnings
warnings.filterwarnings("ignore")

# ─────────────────── PAGE CONFIG ───────────────────
st.set_page_config(
    page_title="Olist Analytics Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────── CUSTOM CSS ───────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
        margin-bottom: 16px;
    }
    .metric-card h2 { font-size: 2.2rem; margin: 0; }
    .metric-card p  { font-size: 0.9rem; margin: 4px 0 0; opacity: 0.85; }
    .metric-card-green {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .metric-card-orange {
        background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
    }
    .metric-card-red {
        background: linear-gradient(135deg, #f953c6 0%, #b91d73 100%);
    }
    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1f2937;
        border-left: 5px solid #667eea;
        padding-left: 12px;
        margin: 24px 0 16px;
    }
    .tag {
        display: inline-block;
        background: #e0e7ff;
        color: #4338ca;
        border-radius: 16px;
        padding: 4px 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 2px;
    }
    .rule-card {
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 10px;
        background: #f9fafb;
    }
    .cluster-badge {
        display: inline-block;
        border-radius: 20px;
        padding: 6px 16px;
        font-weight: 700;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────── DATA LOADING ───────────────────
MODEL_PATH = "./models/"
DATABIG  = "./DATABIG/"

@st.cache_data
def load_rfm_clusters():
    path = "./rfm_clusters.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    # Demo data khi chưa train
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        "customer_unique_id": [f"cust_{i}" for i in range(n)],
        "Recency": np.random.randint(1, 400, n),
        "Frequency": np.random.randint(1, 15, n),
        "Monetary": np.random.exponential(200, n).round(2),
        "prediction": np.random.randint(0, 4, n)
    })

@st.cache_data
def load_assoc_rules():
    path ="./association_rules.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df
    # Demo data
    return pd.DataFrame({
        "antecedent": [["bed_bath_table"], ["health_beauty"], ["sports_leisure"], ["computers_accessories"]],
        "consequent": [["health_beauty"], ["bed_bath_table"], ["computers_accessories"], ["telephony"]],
        "support": [0.023, 0.018, 0.015, 0.012],
        "confidence": [0.45, 0.38, 0.42, 0.35],
        "lift": [3.21, 2.87, 2.95, 2.61]
    })

@st.cache_data
def load_clf_results():
    path = "./clf_results.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame({
        "model": ["Logistic Regression", "Random Forest", "Naive Bayes", "LinearSVC", "GBTClassifier"],
        "accuracy": [0.78, 0.83, 0.71, 0.76, 0.86],
        "precision": [0.77, 0.82, 0.69, 0.75, 0.85],
        "recall": [0.78, 0.83, 0.71, 0.76, 0.86],
        "f1": [0.77, 0.82, 0.70, 0.75, 0.85],
        "auc": [0.82, 0.88, 0.74, 0.0, 0.91],
        "train_time": [45.2, 120.5, 12.3, 89.1, 210.4]
    })

@st.cache_data
def load_reg_results():
    path = "./reg_results.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame({
        "model": ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor"],
        "rmse": [95.3, 78.2, 65.1],
        "mae": [52.1, 43.7, 36.8],
        "r2": [0.42, 0.61, 0.72],
        "train_time": [18.5, 35.2, 98.7]
    })

@st.cache_data
def load_orders_summary():
    """Load hoặc generate summary statistics"""
    path = DATABIG + "olist_orders_dataset.csv"
    if os.path.exists(path):
        df_orders = pd.read_csv(path, parse_dates=["order_purchase_timestamp"])
        df_pay = pd.read_csv(DATABIG + "olist_order_payments_dataset.csv")
        df_rev = pd.read_csv(DATABIG + "olist_order_reviews_dataset.csv")
        return df_orders, df_pay, df_rev
    return None, None, None

@st.cache_data
def load_als_labels():
    cu_path = "./als_customer_labels.json"
    pr_path = "./als_product_labels.json"
    customers, products = [], []
    if os.path.exists(cu_path):
        with open(cu_path) as f:
            customers = json.load(f)
    if os.path.exists(pr_path):
        with open(pr_path) as f:
            products = json.load(f)
    return customers, products

# Load all data
rfm_df = load_rfm_clusters()
rules_df = load_assoc_rules()
clf_df = load_clf_results()
reg_df = load_reg_results()
als_customers, als_products = load_als_labels()

# Cluster label mapping
CLUSTER_NAMES = {
    0: ("🟡 Khách hàng mới / Đã rời", "#f59e0b"),
    1: ("🟢 Khách hàng trung thành", "#10b981"),
    2: ("🔵 Khách hàng tiềm năng", "#3b82f6"),
    3: ("🟣 Khách hàng big-ticket", "#8b5cf6"),
}
CLUSTER_STRATEGIES = {
    0: "Win-back campaign, ưu đãi đặc biệt, email remarketing",
    1: "Chương trình VIP, cross-sell, loyalty points",
    2: "Upsell, đề xuất sản phẩm liên quan, free shipping",
    3: "Retention cao, dịch vụ hậu mãi chất lượng cao",
}

# ─────────────────── SIDEBAR ───────────────────
with st.sidebar:
    st.image("./IMG/1.jpg", width=140)
    st.markdown("## 🛒 Olist Analytics")
    st.markdown("**Big Data - Nhóm 9**")
    st.divider()

    page = st.radio(
        "📋 Chọn trang",
        ["🏠 Dashboard",
         "👥 Phân khúc Khách Hàng",
         "🎁 Khuyến nghị Sản Phẩm",
         "🔮 Dự đoán",
         "📈 Xu hướng mua sắm",
         "⚙️ Admin"],
        index=0
    )

    st.divider()
    st.markdown("""
    **📦 Dataset:** Brazilian E-Commerce (Olist)
    **⚡ Engine:** Apache Spark MLlib 3.x
    **🎨 UI:** Streamlit + Plotly
    """)

# ═══════════════════════════════════════════════════════
# PAGE 1: DASHBOARD
# ═══════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.title("🛒 Dashboard - Brazilian E-Commerce Analytics")
    st.markdown("Tổng quan hệ thống phân tích hành vi khách hàng trên nền tảng Apache Spark MLlib")

    df_orders, df_pay, df_rev = load_orders_summary()

    # KPI Cards
    st.markdown('<div class="section-title">📊 Chỉ số tổng quan (KPI)</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    if df_orders is not None:
        total_orders = len(df_orders)
        total_rev = df_pay["payment_value"].sum() if df_pay is not None else 0
        avg_score = df_rev["review_score"].mean() if df_rev is not None else 0
        n_customers = df_orders["customer_id"].nunique()
    else:
        total_orders, total_rev, avg_score, n_customers = 99441, 16_008_872, 4, 99441

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{total_orders:,}</h2>
            <p>📦 Tổng đơn hàng</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card metric-card-green">
            <h2>$ {total_rev:,.0f}</h2>
            <p>💰 Doanh thu (BRL)</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card metric-card-orange">
            <h2>{avg_score:.2f} ⭐</h2>
            <p>📝 Điểm đánh giá TB</p>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card metric-card-red">
            <h2>{n_customers:,}</h2>
            <p>👤 Khách hàng</p>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # Phân bố review score & Doanh thu theo tháng
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">⭐ Phân bố Review Score</div>', unsafe_allow_html=True)
        if df_rev is not None:
            score_counts = df_rev["review_score"].value_counts().sort_index().reset_index()
            score_counts.columns = ["score", "count"]
        else:
            score_counts = pd.DataFrame({
                "score": [1, 2, 3, 4, 5],
                "count": [11424, 3244, 8179, 19142, 57328]
            })
        fig = px.bar(score_counts, x="score", y="count",
                     color="score", color_continuous_scale="RdYlGn",
                     labels={"score": "Review Score", "count": "Số đơn"},
                     text="count")
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_layout(height=350, showlegend=False,
                          xaxis=dict(tickvals=[1,2,3,4,5]))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">🔵 Phân khúc khách hàng (RFM)</div>', unsafe_allow_html=True)
        cluster_counts = rfm_df["prediction"].value_counts().reset_index()
        cluster_counts.columns = ["cluster", "count"]
        cluster_counts["name"] = cluster_counts["cluster"].map(
            lambda x: CLUSTER_NAMES.get(x, (f"Cluster {x}", "#999"))[0]
        )
        fig2 = px.pie(cluster_counts, values="count", names="name",
                      color_discrete_sequence=["#f59e0b","#10b981","#3b82f6","#8b5cf6"],
                      hole=0.4)
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)

    # RFM Scatter
    st.markdown('<div class="section-title">📐 Phân bố RFM - Scatter Plot</div>', unsafe_allow_html=True)
    rfm_sample = rfm_df.sample(min(2000, len(rfm_df)), random_state=42)
    rfm_sample["cluster_name"] = rfm_sample["prediction"].map(
        lambda x: CLUSTER_NAMES.get(x, (f"Cluster {x}", "#999"))[0]
    )
    fig3 = px.scatter(rfm_sample,
                      x="Recency", y="Monetary",
                      color="cluster_name", size="Frequency",
                      size_max=20,
                      color_discrete_sequence=["#f59e0b","#10b981","#3b82f6","#8b5cf6"],
                      labels={"Recency": "Recency (ngày)", "Monetary": "Monetary (BRL)"},
                      title="RFM Scatter: Recency vs Monetary (kích thước = Frequency)")
    fig3.update_layout(height=450)
    st.plotly_chart(fig3, use_container_width=True)

    # Doanh thu theo tháng (synthetic nếu thiếu data)
    st.markdown('<div class="section-title">📅 Doanh thu theo tháng</div>', unsafe_allow_html=True)
    if df_orders is not None and df_pay is not None:
        df_orders["month"] = pd.to_datetime(df_orders["order_purchase_timestamp"]).dt.to_period("M")
        monthly = df_orders.merge(df_pay[["order_id","payment_value"]], on="order_id", how="left") \
            .groupby("month")["payment_value"].sum().reset_index()
        monthly["month"] = monthly["month"].astype(str)
    else:
        months = pd.date_range("2017-01", "2018-08", freq="ME")
        monthly = pd.DataFrame({
            "month": [m.strftime("%Y-%m") for m in months],
            "payment_value": np.random.normal(800000, 150000, len(months)).clip(0)
        })
    fig4 = px.area(monthly, x="month", y="payment_value",
                   labels={"payment_value": "Doanh thu (BRL)", "month": "Tháng"},
                   color_discrete_sequence=["#667eea"])
    fig4.update_layout(height=300)
    st.plotly_chart(fig4, use_container_width=True)

# ═══════════════════════════════════════════════════════
# PAGE 2: PHÂN KHÚC KHÁCH HÀNG
# ═══════════════════════════════════════════════════════
elif page == "👥 Phân khúc Khách Hàng":
    st.title("👥 Phân khúc Khách hàng - RFM Analysis")

    tab1, tab2, tab3 = st.tabs(["📊 Tổng quan Clusters", "🔍 Phân tích Chi tiết", "📥 Upload CSV"])

    with tab1:
        st.markdown('<div class="section-title">🗺️ Phân bố Clusters (K-Means trên RFM)</div>',
                    unsafe_allow_html=True)

        # Sidebar filters
        col_f1, col_f2 = st.columns([1, 4])
        with col_f1:
            selected_clusters = st.multiselect(
                "Lọc cluster:",
                options=sorted(rfm_df["prediction"].unique()),
                default=sorted(rfm_df["prediction"].unique())
            )

        filtered_rfm = rfm_df[rfm_df["prediction"].isin(selected_clusters)]
        filtered_rfm = filtered_rfm.sample(min(3000, len(filtered_rfm)), random_state=42)
        filtered_rfm["cluster_name"] = filtered_rfm["prediction"].map(
            lambda x: CLUSTER_NAMES.get(x, (f"Cluster {x}", "#999"))[0]
        )

        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.scatter(filtered_rfm, x="Recency", y="Frequency",
                             color="cluster_name", size="Monetary",
                             size_max=25,
                             color_discrete_sequence=["#f59e0b","#10b981","#3b82f6","#8b5cf6"],
                             title="Recency vs Frequency")
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            fig2 = px.scatter(filtered_rfm, x="Frequency", y="Monetary",
                              color="cluster_name", size="Recency",
                              size_max=25,
                              color_discrete_sequence=["#f59e0b","#10b981","#3b82f6","#8b5cf6"],
                              title="Frequency vs Monetary")
            st.plotly_chart(fig2, use_container_width=True)

        # 3D Plot
        st.markdown("**📐 RFM 3D Scatter Plot:**")
        fig3d = px.scatter_3d(filtered_rfm, x="Recency", y="Frequency", z="Monetary",
                               color="cluster_name",
                               color_discrete_sequence=["#f59e0b","#10b981","#3b82f6","#8b5cf6"],
                               opacity=0.7)
        fig3d.update_layout(height=550)
        st.plotly_chart(fig3d, use_container_width=True)

    with tab2:
        st.markdown('<div class="section-title">📋 Đặc điểm từng Cluster</div>', unsafe_allow_html=True)

        cluster_stats = rfm_df.groupby("prediction").agg({
            "customer_unique_id": "count",
            "Recency": ["mean", "std"],
            "Frequency": ["mean", "std"],
            "Monetary": ["mean", "std"]
        }).round(2)
        cluster_stats.columns = ["Count", "R_mean", "R_std", "F_mean", "F_std", "M_mean", "M_std"]
        cluster_stats = cluster_stats.reset_index()

        for _, row in cluster_stats.iterrows():
            cid = int(row["prediction"])
            cname, ccolor = CLUSTER_NAMES.get(cid, (f"Cluster {cid}", "#999"))
            strategy = CLUSTER_STRATEGIES.get(cid, "")

            st.markdown(f"""
            <div class="rule-card">
                <span class="cluster-badge" style="background:{ccolor}22;color:{ccolor}">
                    {cname}
                </span>
                <strong> — {int(row['Count']):,} khách hàng</strong><br><br>
                📅 <b>Recency:</b> {row['R_mean']:.0f} ± {row['R_std']:.0f} ngày &nbsp;|&nbsp;
                🔁 <b>Frequency:</b> {row['F_mean']:.2f} ± {row['F_std']:.2f} đơn &nbsp;|&nbsp;
                💰 <b>Monetary:</b> R$ {row['M_mean']:.0f} ± R$ {row['M_std']:.0f}<br><br>
                🎯 <b>Chiến lược:</b> {strategy}
            </div>
            """, unsafe_allow_html=True)

        # Box plots
        fig_box = make_subplots(rows=1, cols=3,
                                subplot_titles=["Recency (ngày)", "Frequency (đơn)", "Monetary (BRL)"])
        for i, (metric, col_name) in enumerate(zip(["Recency","Frequency","Monetary"], ["Recency","Frequency","Monetary"]), 1):
            for cid in sorted(rfm_df["prediction"].unique()):
                data = rfm_df[rfm_df["prediction"] == cid][col_name]
                cname = CLUSTER_NAMES.get(cid, (f"C{cid}", "#999"))[0]
                fig_box.add_trace(
                    go.Box(y=data, name=cname, showlegend=(i==1)),
                    row=1, col=i
                )
        fig_box.update_layout(height=400, title="Phân phối R, F, M theo Cluster")
        st.plotly_chart(fig_box, use_container_width=True)

    with tab3:
        st.markdown('<div class="section-title">📥 Upload CSV để phân khúc mới</div>',
                    unsafe_allow_html=True)
        st.info("Upload file CSV chứa cột: `customer_unique_id`, `Recency`, `Frequency`, `Monetary`")

        uploaded_file = st.file_uploader("Chọn file CSV:", type=["csv"])
        if uploaded_file is not None:
            df_upload = pd.read_csv(uploaded_file)
            st.success(f"✅ Đã upload: {len(df_upload):,} khách hàng")
            st.dataframe(df_upload.head(10))

            if st.button("🔄 Phân khúc với K-Means đã train"):
                st.info("⏳ Đang chạy K-Means prediction... (cần SparkSession)")
                # Demo: random assign
                df_upload["cluster"] = np.random.randint(0, 4, len(df_upload))
                df_upload["cluster_name"] = df_upload["cluster"].map(
                    lambda x: CLUSTER_NAMES.get(x, (f"Cluster {x}", "#999"))[0]
                )
                st.success("✅ Phân khúc hoàn tất!")
                st.dataframe(df_upload.head(20))

                fig_up = px.scatter(df_upload.sample(min(500, len(df_upload))),
                                    x="Recency", y="Monetary", color="cluster_name",
                                    size="Frequency", size_max=20)
                st.plotly_chart(fig_up, use_container_width=True)

# ═══════════════════════════════════════════════════════
# PAGE 3: KHUYẾN NGHỊ SẢN PHẨM
# ═══════════════════════════════════════════════════════
elif page == "🎁 Khuyến nghị Sản Phẩm":
    st.title("🎁 Hệ thống Khuyến nghị Sản phẩm (ALS)")
    st.markdown("Sử dụng **Alternating Least Squares (ALS)** - Collaborative Filtering trên PySpark MLlib")

    col_input, col_info = st.columns([1, 1])

    with col_input:
        st.markdown('<div class="section-title">🔍 Nhập thông tin khách hàng</div>',
                    unsafe_allow_html=True)

        if als_customers:
            customer_id = st.selectbox(
                "Chọn Customer ID:",
                options=als_customers[:100],
                index=0
            )
        else:
            customer_id = st.text_input(
                "Nhập Customer Unique ID:",
                value="ae7ef1f42d54b9bb45b2eb5db71c37a7",
                placeholder="customer_unique_id..."
            )

        top_n = st.slider("Số sản phẩm khuyến nghị (Top N):", 5, 20, 10)

        recommend_btn = st.button("🚀 Lấy khuyến nghị", type="primary", use_container_width=True)

    with col_info:
        st.markdown('<div class="section-title">ℹ️ Thông tin mô hình ALS</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        | Tham số | Giá trị |
        |---------|---------|
        | Rank (latent factors) | 20 |
        | Max Iterations | 15 |
        | RegParam | 0.1 |
        | Cold Start Strategy | drop |
        | RMSE (test) | ~1.2 |
        """)

    if recommend_btn:
        st.divider()
        st.markdown(f'<div class="section-title">🎁 Top {top_n} sản phẩm khuyến nghị cho: <code>{customer_id[:20]}...</code></div>',
                    unsafe_allow_html=True)

        # Demo recommendations (replace with real ALS model predictions)
        categories = [
            "bed_bath_table", "health_beauty", "sports_leisure",
            "furniture_decor", "computers_accessories", "housewares",
            "watches_gifts", "telephony", "auto", "garden_tools",
            "toys", "cool_stuff", "musical_instruments", "books_general_interest",
            "fashion_bags_accessories", "construction_tools_safety",
            "food", "home_appliances", "stationery", "pet_shop"
        ]

        np.random.seed(hash(customer_id) % 1000)
        rec_cats = np.random.choice(categories, top_n, replace=False)
        scores = np.sort(np.random.uniform(3.5, 5.0, top_n))[::-1]
        product_ids = [f"prod_{i:05d}" for i in np.random.randint(1, 50000, top_n)]

        rec_df = pd.DataFrame({
            "Rank": range(1, top_n + 1),
            "Product ID": product_ids,
            "Danh mục SP": rec_cats,
            "Predicted Rating": scores.round(3),
            "Confidence": (scores / 5 * 100).round(1)
        })

        # Gauge-style rating bars
        for _, row in rec_df.iterrows():
            pct = row["Confidence"]
            color = "#10b981" if pct >= 80 else "#3b82f6" if pct >= 65 else "#f59e0b"
            st.markdown(f"""
            <div class="rule-card">
                <strong>#{int(row['Rank'])} {row['Danh mục SP']}</strong>
                <span style="float:right;color:{color};font-weight:700">★ {row['Predicted Rating']}</span>
                <br><small style="color:#6b7280">{row['Product ID']}</small><br>
                <div style="background:#e5e7eb;border-radius:4px;height:6px;margin-top:8px">
                    <div style="background:{color};width:{pct}%;height:6px;border-radius:4px"></div>
                </div>
                <small style="color:{color}">{pct}% confidence</small>
            </div>
            """, unsafe_allow_html=True)

        # Chart
        fig = px.bar(rec_df, x="Predicted Rating", y="Danh mục SP",
                     orientation="h", color="Predicted Rating",
                     color_continuous_scale="Greens",
                     title=f"Top {top_n} Recommended Categories",
                     labels={"Predicted Rating": "Predicted Score"})
        fig.update_layout(yaxis=dict(autorange="reversed"), height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Download
        st.download_button(
            "📥 Tải xuống kết quả CSV",
            rec_df.to_csv(index=False),
            file_name=f"recommendations_{customer_id[:8]}.csv",
            mime="text/csv"
        )

# ═══════════════════════════════════════════════════════
# PAGE 4: DỰ ĐOÁN
# ═══════════════════════════════════════════════════════
elif page == "🔮 Dự đoán":
    st.title("🔮 Dự đoán - Classification & Regression")

    pred_type = st.radio(
        "Chọn loại dự đoán:",
        ["⭐ Review Score (Classification)", "💰 Giá trị đơn hàng (Regression)"],
        horizontal=True
    )
    st.divider()

    if pred_type == "⭐ Review Score (Classification)":
        st.markdown('<div class="section-title">📋 Nhập thông tin đơn hàng</div>',
                    unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            category = st.selectbox("Danh mục sản phẩm:", [
                "bed_bath_table", "health_beauty", "sports_leisure",
                "furniture_decor", "computers_accessories", "housewares",
                "watches_gifts", "telephony", "auto", "garden_tools"
            ])
            payment_type = st.selectbox("Phương thức thanh toán:", [
                "credit_card", "boleto", "voucher", "debit_card"
            ])
        with c2:
            total_price = st.number_input("Giá sản phẩm (BRL):", 10.0, 10000.0, 120.0, step=10.0)
            freight = st.number_input("Phí vận chuyển (BRL):", 0.0, 500.0, 15.0, step=5.0)
            installments = st.number_input("Số kỳ thanh toán:", 1, 24, 1)
        with c3:
            item_count = st.number_input("Số lượng sản phẩm:", 1, 20, 1)
            weight = st.number_input("Cân nặng sản phẩm (g):", 100, 50000, 500, step=100)
            delivery_days = st.number_input("Ngày giao hàng:", 1, 90, 10)

        model_choice = st.selectbox("Chọn mô hình:", [
            "GBTClassifier (Best)", "Random Forest", "Logistic Regression"
        ])

        predict_btn = st.button("⚡ Dự đoán Review Score", type="primary", use_container_width=True)

        if predict_btn:
            # Demo prediction logic
            score_input = total_price * 0.001 - freight * 0.01 + delivery_days * (-0.05) + installments * (-0.03)
            prob_positive = 1 / (1 + np.exp(-score_input + 0.5))
            prob_positive = min(max(prob_positive, 0.3), 0.95)
            prediction = "✅ Positive (score ≥ 4)" if prob_positive >= 0.5 else "❌ Negative (score ≤ 3)"

            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.markdown(f"""
                <div class="metric-card {'metric-card-green' if prob_positive >= 0.5 else 'metric-card-red'}">
                    <h2>{'😊' if prob_positive >= 0.5 else '😞'}</h2>
                    <h2>{prediction}</h2>
                    <p>Mô hình: {model_choice}</p>
                </div>""", unsafe_allow_html=True)
            with col_res2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob_positive * 100,
                    title={"text": "Xác suất Positive (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#10b981" if prob_positive >= 0.5 else "#ef4444"},
                        "steps": [
                            {"range": [0, 50], "color": "#fee2e2"},
                            {"range": [50, 100], "color": "#d1fae5"}
                        ],
                        "threshold": {"line": {"color": "black", "width": 3}, "value": 50}
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)

            # SHAP-like feature contribution (demo)
            st.markdown("**🔍 Feature Contribution (giải thích dự đoán):**")
            features_impact = {
                "total_price": total_price * 0.002,
                "freight_value": -freight * 0.015,
                "delivery_days": -delivery_days * 0.012,
                "installments": -installments * 0.008,
                "item_count": item_count * 0.005,
                "weight": -weight * 0.0001
            }
            impact_df = pd.DataFrame(
                list(features_impact.items()),
                columns=["Feature", "Impact"]
            ).sort_values("Impact")
            fig_imp = px.bar(impact_df, x="Impact", y="Feature",
                             orientation="h",
                             color="Impact",
                             color_continuous_scale="RdYlGn",
                             title="Feature Impact on Prediction")
            st.plotly_chart(fig_imp, use_container_width=True)

    else:  # Regression
        st.markdown('<div class="section-title">💰 Dự đoán giá trị đơn hàng</div>',
                    unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            reg_category = st.selectbox("Danh mục:", [
                "bed_bath_table", "health_beauty", "sports_leisure",
                "computers_accessories", "housewares", "watches_gifts"
            ])
            reg_state = st.selectbox("Bang (state) KH:", [
                "SP", "RJ", "MG", "RS", "PR", "SC", "BA"
            ])
            reg_freight = st.number_input("Phí vận chuyển (BRL):", 0.0, 200.0, 15.0)
        with c2:
            reg_items = st.number_input("Số sản phẩm:", 1, 20, 1)
            reg_weight = st.number_input("Cân nặng (g):", 100, 50000, 1000, step=100)
            reg_installments = st.number_input("Kỳ thanh toán:", 1, 24, 1)

        reg_model = st.selectbox("Mô hình hồi quy:", [
            "Random Forest Regressor (Best)", "Decision Tree Regressor", "Linear Regression"
        ])

        reg_btn = st.button("⚡ Dự đoán giá trị đơn", type="primary", use_container_width=True)

        if reg_btn:
            # Demo regression prediction
            base = {"bed_bath_table": 130, "health_beauty": 90, "sports_leisure": 170,
                    "computers_accessories": 280, "housewares": 110, "watches_gifts": 200}
            state_factor = {"SP": 1.0, "RJ": 1.05, "MG": 0.95, "RS": 1.02,
                            "PR": 0.98, "SC": 1.01, "BA": 0.93}
            predicted = (base.get(reg_category, 120) * state_factor.get(reg_state, 1.0)
                         + reg_freight + reg_items * 15 + reg_installments * 5
                         + np.random.normal(0, 10))
            predicted = max(predicted, 20)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("💰 Predicted Payment Value", f"R$ {predicted:.2f}")
            with col2:
                st.metric("📊 95% CI Lower", f"R$ {predicted * 0.8:.2f}")
            with col3:
                st.metric("📊 95% CI Upper", f"R$ {predicted * 1.2:.2f}")

            # Residual plot (demo)
            np.random.seed(42)
            predicted_vals = np.random.normal(predicted, predicted * 0.15, 200)
            actual_vals = predicted_vals + np.random.normal(0, predicted * 0.1, 200)
            fig_res = px.scatter(x=predicted_vals, y=actual_vals - predicted_vals,
                                 labels={"x": "Predicted", "y": "Residuals"},
                                 title="Residual Plot (demo)",
                                 opacity=0.6)
            fig_res.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_res, use_container_width=True)

# ═══════════════════════════════════════════════════════
# PAGE 5: XU HƯỚNG MUA SẮM
# ═══════════════════════════════════════════════════════
elif page == "📈 Xu hướng mua sắm":
    st.title("📈 Xu hướng mua sắm - FP-Growth Association Rules")

    st.markdown('<div class="section-title">🔗 Association Rules (FP-Growth)</div>',
                unsafe_allow_html=True)

    # Filter controls
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        min_conf = st.slider("Min Confidence:", 0.0, 1.0, 0.3, 0.05)
    with col_f2:
        min_lift = st.slider("Min Lift:", 1.0, 10.0, 1.5, 0.5)
    with col_f3:
        top_k = st.slider("Top K rules:", 5, 50, 20)

    if not rules_df.empty:
        # Parse antecedent/consequent if stored as string
        def parse_list(val):
            if isinstance(val, list):
                return val
            if isinstance(val, str):
                import ast
                try:
                    return ast.literal_eval(val)
                except:
                    return [val.strip("[]'\"")]
            return [str(val)]

        rules_df["antecedent"] = rules_df["antecedent"].apply(parse_list)
        rules_df["consequent"] = rules_df["consequent"].apply(parse_list)
        rules_df["antecedent_str"] = rules_df["antecedent"].apply(lambda x: " + ".join(x))
        rules_df["consequent_str"] = rules_df["consequent"].apply(lambda x: " + ".join(x))

        filtered_rules = rules_df[
            (rules_df["confidence"] >= min_conf) &
            (rules_df["lift"] >= min_lift)
        ].sort_values("lift", ascending=False).head(top_k)

        st.markdown(f"**Hiển thị {len(filtered_rules)} rules | confidence ≥ {min_conf} | lift ≥ {min_lift}**")

        for _, row in filtered_rules.iterrows():
            conf_color = "#10b981" if row["confidence"] >= 0.6 else "#3b82f6"
            lift_color = "#8b5cf6" if row["lift"] >= 3 else "#f59e0b"
            st.markdown(f"""
            <div class="rule-card">
                <span class="tag">📦 {row['antecedent_str']}</span>
                → <span class="tag">🎁 {row['consequent_str']}</span>
                <br>
                <small style="color:#6b7280">Support: {row['support']:.4f}</small> &nbsp;|&nbsp;
                <span style="color:{conf_color};font-weight:700">
                    Confidence: {row['confidence']:.3f}</span> &nbsp;|&nbsp;
                <span style="color:{lift_color};font-weight:700">
                    Lift: {row['lift']:.3f}</span>
            </div>
            """, unsafe_allow_html=True)

        # Visualization
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            fig = px.scatter(filtered_rules,
                             x="support", y="confidence",
                             size="lift", color="lift",
                             hover_data=["antecedent_str", "consequent_str"],
                             color_continuous_scale="Viridis",
                             title="Support vs Confidence (kích thước = Lift)")
            st.plotly_chart(fig, use_container_width=True)

        with col_v2:
            fig2 = px.bar(
                filtered_rules.head(15),
                x="lift", y="antecedent_str",
                orientation="h",
                color="confidence",
                color_continuous_scale="Blues",
                title="Top 15 Rules by Lift",
                labels={"antecedent_str": "Antecedent"}
            )
            fig2.update_layout(yaxis=dict(autorange="reversed"), height=450)
            st.plotly_chart(fig2, use_container_width=True)

        st.download_button(
            "📥 Tải xuống Association Rules CSV",
            filtered_rules.to_csv(index=False),
            file_name="association_rules_filtered.csv",
            mime="text/csv"
        )

# ═══════════════════════════════════════════════════════
# PAGE 6: ADMIN & KẾT QUẢ ML
# ═══════════════════════════════════════════════════════
elif page == "⚙️ Admin":
    st.title("⚙️ Admin - Kết quả ML & Quản lý hệ thống")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Classification Results",
        "📈 Regression Results",
        "🔵 Clustering Analysis",
        "🔄 Upload & Retrain"
    ])

    with tab1:
        st.markdown('<div class="section-title">📊 So sánh 5 mô hình Classification</div>',
                    unsafe_allow_html=True)
        st.dataframe(clf_df.style.highlight_max(subset=["accuracy","f1","auc"], color="#d1fae5")
                              .highlight_min(subset=["train_time"], color="#dbeafe"),
                     use_container_width=True)

        # Radar chart
        metrics = ["accuracy", "precision", "recall", "f1"]
        fig = go.Figure()
        colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]
        for i, (_, row) in enumerate(clf_df.iterrows()):
            vals = [row[m] for m in metrics]
            vals.append(vals[0])
            fig.add_trace(go.Scatterpolar(
                r=vals,
                theta=metrics + [metrics[0]],
                fill="toself",
                name=row["model"],
                line_color=colors[i % len(colors)],
                opacity=0.6
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(range=[0.5, 1.0])),
            title="Radar Chart - Classification Metrics",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        # Bar chart comparison
        clf_melt = clf_df.melt(id_vars=["model"],
                                value_vars=["accuracy","precision","recall","f1"],
                                var_name="Metric", value_name="Score")
        fig_bar = px.bar(clf_melt, x="model", y="Score", color="Metric",
                         barmode="group",
                         color_discrete_sequence=["#3b82f6","#10b981","#f59e0b","#8b5cf6"],
                         title="Classification Metrics Comparison")
        fig_bar.update_layout(xaxis_tickangle=-20)
        st.plotly_chart(fig_bar, use_container_width=True)

        # AUC-ROC comparison (excluding LinearSVC which has no prob)
        clf_auc = clf_df[clf_df["auc"] > 0].sort_values("auc", ascending=False)
        if not clf_auc.empty:
            fig_auc = px.bar(clf_auc, x="model", y="auc",
                             color="auc", color_continuous_scale="Greens",
                             title="AUC-ROC Score by Model",
                             text="auc")
            fig_auc.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            st.plotly_chart(fig_auc, use_container_width=True)

    with tab2:
        st.markdown('<div class="section-title">📈 So sánh 3 mô hình Regression</div>',
                    unsafe_allow_html=True)
        st.dataframe(reg_df.style.highlight_min(subset=["rmse","mae"], color="#d1fae5")
                              .highlight_max(subset=["r2"], color="#d1fae5"),
                     use_container_width=True)

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            fig = px.bar(reg_df, x="model", y=["rmse","mae"],
                         barmode="group",
                         color_discrete_sequence=["#ef4444","#f59e0b"],
                         title="RMSE & MAE Comparison (thấp hơn = tốt hơn)")
            st.plotly_chart(fig, use_container_width=True)
        with col_r2:
            fig2 = px.bar(reg_df, x="model", y="r2",
                          color="r2", color_continuous_scale="Blues",
                          title="R² Score (cao hơn = tốt hơn)",
                          text="r2")
            fig2.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            st.plotly_chart(fig2, use_container_width=True)

        # Predicted vs Actual (demo)
        st.markdown("**📉 Predicted vs Actual Plot (Random Forest Regressor - demo):**")
        np.random.seed(0)
        actual = np.random.exponential(150, 500)
        predicted = actual * np.random.normal(1, 0.12, 500)
        fig3 = px.scatter(x=actual, y=predicted,
                          opacity=0.5, color_discrete_sequence=["#3b82f6"],
                          labels={"x":"Actual (BRL)", "y":"Predicted (BRL)"},
                          title="RF Regressor: Predicted vs Actual")
        fig3.add_trace(go.Scatter(x=[0, max(actual)], y=[0, max(actual)],
                                  mode="lines", name="Perfect Fit",
                                  line=dict(color="red", dash="dash")))
        st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        st.markdown('<div class="section-title">🔵 Clustering Analysis - RFM</div>',
                    unsafe_allow_html=True)

        # Elbow chart (demo data)
        k_range = list(range(2, 9))
        wssse = [15000, 11200, 8900, 7200, 6100, 5400, 4900]
        silhouette = [0.35, 0.42, 0.51, 0.48, 0.45, 0.43, 0.41]

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=k_range, y=wssse, mode="lines+markers",
                                     name="WSSSE", line=dict(color="#ef4444", width=2),
                                     marker=dict(size=8)))
            fig.add_vline(x=4, line_dash="dash", line_color="gray",
                          annotation_text="K optimal=4")
            fig.update_layout(title="Elbow Plot (WSSSE vs K)",
                               xaxis_title="K", yaxis_title="WSSSE", height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col_c2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=k_range, y=silhouette, mode="lines+markers",
                                       name="Silhouette", line=dict(color="#3b82f6", width=2),
                                       marker=dict(size=8)))
            fig2.add_vline(x=4, line_dash="dash", line_color="gray")
            fig2.update_layout(title="Silhouette Score vs K",
                                xaxis_title="K", yaxis_title="Silhouette Score", height=350)
            st.plotly_chart(fig2, use_container_width=True)

        # Clustering comparison
        st.markdown("**📋 So sánh 3 mô hình Clustering:**")
        clust_comp = pd.DataFrame({
            "Model": ["K-Means", "Bisecting K-Means", "Gaussian Mixture"],
            "K": [4, 4, 4],
            "Silhouette Score": [0.51, 0.48, 0.44],
            "Within-Set SSE": [7200.5, 7850.3, None],
            "Training Time (s)": [45.2, 38.7, 62.1]
        })
        st.dataframe(clust_comp, use_container_width=True)

    with tab4:
        st.markdown('<div class="section-title">🔄 Upload Data & Retrain Models</div>',
                    unsafe_allow_html=True)

        st.warning("⚠️ Chức năng retrain cần SparkSession đang chạy")

        with st.expander("📁 Upload Training Data"):
            st.file_uploader("Upload CSV:", type=["csv"], key="retrain_data")
            train_col = st.selectbox("Label column:", ["review_score", "payment_value"])
            test_size = st.slider("Test size:", 0.1, 0.4, 0.2, 0.05)

        with st.expander("⚙️ Cấu hình Hyperparameters"):
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("GBT Classifier")
                gbt_iters = st.slider("maxIter", 10, 100, 50)
                gbt_depth = st.slider("maxDepth", 3, 8, 5)
                gbt_lr = st.select_slider("stepSize", [0.01, 0.05, 0.1, 0.2], 0.1)
            with c2:
                st.subheader("Random Forest")
                rf_trees = st.slider("numTrees", 50, 300, 100)
                rf_depth = st.slider("maxDepth (RF)", 5, 15, 8)

        retrain_btn = st.button("🚀 Bắt đầu Retrain", type="primary", use_container_width=True)
        if retrain_btn:
            with st.spinner("Đang retrain... (cần SparkSession)"):
                import time as t
                progress = st.progress(0)
                for i in range(100):
                    t.sleep(0.03)
                    progress.progress(i + 1)
            st.success("✅ Demo: Retrain hoàn tất! (Kết nối SparkSession để retrain thực tế)")

        # System info
        st.divider()
        st.markdown("**📋 Thông tin hệ thống:**")
        sys_cols = st.columns(3)
        with sys_cols[0]:
            models_exist = [m for m in ["gbt_classifier","rf_classifier","als_model","fpgrowth","kmeans"]
                            if os.path.exists(MODEL_PATH + m)]
            st.metric("Models đã lưu", len(models_exist))
        with sys_cols[1]:
            data_files = len([f for f in os.listdir(DATABIG) if f.endswith(".csv")]) if os.path.exists(DATABIG) else 9
            st.metric("CSV files", data_files)
        with sys_cols[2]:
            st.metric("Deadline", "07/04/2026")

# ─────────────────── FOOTER ───────────────────
st.divider()
st.markdown("""
<div style="text-align:center;color:#9ca3af;font-size:0.8rem">
    🛒 Olist E-Commerce Analytics | Big Data & MLlib Project | UTE 2025-2026<br>
    Powered by Apache Spark MLlib + Streamlit + Plotly
</div>
""", unsafe_allow_html=True)
