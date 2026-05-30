import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Dashboard DA6", layout="wide")

st.title("Dashboard для аналізу компаній")
st.write("Додаток дозволяє фільтрувати дані, переглядати таблицю, будувати графіки та виконувати кластеризацію компаній.")

df = pd.read_csv("streamlit_dataset.csv")

st.sidebar.header("Панель фільтрації")
st.sidebar.write("Оберіть параметри для аналізу даних.")

region = st.sidebar.selectbox("Оберіть регіон", sorted(df["Region"].unique()))

industries = st.sidebar.multiselect(
    "Оберіть галузі",
    sorted(df["Industry"].unique()),
    default=sorted(df["Industry"].unique())
)

scenario = st.sidebar.radio(
    "Оберіть сценарій",
    sorted(df["Scenario"].unique())
)

show_map = st.sidebar.checkbox("Показати карту компаній", value=True)

filtered_df = df[
    (df["Region"] == region) &
    (df["Industry"].isin(industries)) &
    (df["Scenario"] == scenario)
]

st.subheader("Відфільтровані дані")

selected_columns = st.multiselect(
    "Оберіть колонки для відображення",
    filtered_df.columns,
    default=["Company", "Year", "Region", "Industry", "Revenue", "Expenses", "Profit", "ROI"]
)

st.dataframe(filtered_df[selected_columns], use_container_width=True)

st.subheader("Основні показники")

col1, col2, col3 = st.columns(3)

col1.metric("Кількість компаній", len(filtered_df))
col2.metric("Середній прибуток", round(filtered_df["Profit"].mean(), 2) if len(filtered_df) > 0 else 0)
col3.metric("Середній ROI", round(filtered_df["ROI"].mean(), 2) if len(filtered_df) > 0 else 0)

st.subheader("Графічне представлення даних")

if len(filtered_df) > 0:
    fig1 = px.bar(
        filtered_df,
        x="Company",
        y="Profit",
        color="Industry",
        title="Прибуток компаній за галузями"
    )
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(
        filtered_df,
        x="Revenue",
        y="Expenses",
        size="Customers",
        color="Industry",
        hover_name="Company",
        title="Доходи та витрати компаній"
    )
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.histogram(
        filtered_df,
        x="ROI",
        color="Industry",
        title="Розподіл ROI"
    )
    st.plotly_chart(fig3, use_container_width=True)

    if show_map:
        st.subheader("Карта компаній")
        fig4 = px.scatter_mapbox(
            filtered_df,
            lat="Latitude",
            lon="Longitude",
            hover_name="Company",
            hover_data=["Region", "Industry", "Profit"],
            zoom=4,
            height=500,
            title="Географічне розташування компаній"
        )
        fig4.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig4, use_container_width=True)
else:
    st.warning("Немає даних для обраних фільтрів.")

st.subheader("Кластеризація компаній методом KMeans")

cluster_columns = ["Revenue", "Expenses", "Investment", "Customers", "Profit", "ROI", "AdBudget"]

if len(filtered_df) >= 3:
    n_clusters = st.slider("Оберіть кількість кластерів", 2, 5, 3)

    X = filtered_df[cluster_columns]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    filtered_df = filtered_df.copy()
    filtered_df["Cluster"] = kmeans.fit_predict(X_scaled)

    fig5 = px.scatter(
        filtered_df,
        x="Revenue",
        y="Profit",
        color="Cluster",
        hover_name="Company",
        title="Кластеризація компаній за доходом і прибутком"
    )
    st.plotly_chart(fig5, use_container_width=True)

    st.write("Таблиця з результатами кластеризації:")
    st.dataframe(filtered_df[["Company", "Revenue", "Profit", "ROI", "Cluster"]], use_container_width=True)
else:
    st.info("Для кластеризації потрібно мінімум 3 записи після фільтрації.")

st.subheader("Регресійна модель")

numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

x_col = st.selectbox("Оберіть змінну X", numeric_columns, index=numeric_columns.index("Expenses"))
y_col = st.selectbox("Оберіть змінну Y", numeric_columns, index=numeric_columns.index("Profit"))

if len(filtered_df) > 1:
    X_reg = filtered_df[[x_col]]
    y_reg = filtered_df[y_col]

    model = LinearRegression()
    model.fit(X_reg, y_reg)

    r2 = model.score(X_reg, y_reg)

    st.write(f"Коефіцієнт моделі: {model.coef_[0]:.4f}")
    st.write(f"Вільний член: {model.intercept_:.4f}")
    st.write(f"Якість моделі R²: {r2:.4f}")

    fig6 = px.scatter(
        filtered_df,
        x=x_col,
        y=y_col,
        trendline="ols",
        title=f"Регресійна модель: {y_col} залежно від {x_col}"
    )
    st.plotly_chart(fig6, use_container_width=True)
else:
    st.info("Для регресійної моделі потрібно більше одного запису.")