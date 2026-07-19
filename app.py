# ============================================================
# VIGIL — Virus Intelligence & Global Intervention Laboratory
# Streamlit Dashboard
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from supabase import create_client
import folium
from streamlit_folium import folium_static
from datetime import datetime

# ============================================================
# НАСТРОЙКА СТРАНИЦЫ
# ============================================================

st.set_page_config(
    page_title="VIGIL — Global Virus Surveillance",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# ПОДКЛЮЧЕНИЕ К SUPABASE
# ============================================================

SUPABASE_URL = "https://wdhsqybkzpnqrfnvgiwb.supabase.co"
SUPABASE_KEY = "sb_publishable_D4j5tNHVhW1gf-orm3kNwg_f-pqfxso"

@st.cache_data(ttl=3600)
def load_data():
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    response = supabase.table('outbreaks').select('*').execute()
    return pd.DataFrame(response.data)

df = load_data()

# ============================================================
# БОКОВАЯ ПАНЕЛЬ
# ============================================================

with st.sidebar:
    st.image("assets/logo.png", width=200) if False else st.markdown("## 🦠 VIGIL")
    st.title("VIGIL")
    st.caption("Virus Intelligence & Global Intervention Laboratory")
    st.markdown("---")
    
    # Фильтры
    years = sorted(df['year'].unique())
    selected_year = st.slider(
        "Выберите год",
        min_value=min(years),
        max_value=max(years),
        value=max(years)
    )
    
    countries = ['Все'] + sorted(df['country'].unique())
    selected_country = st.selectbox("Страна", countries)
    
    st.markdown("---")
    st.caption("Science for Future")
    st.caption(f"Данные обновлены: {datetime.now().strftime('%d.%m.%Y %H:%M')}")

# ============================================================
# ФИЛЬТРАЦИЯ
# ============================================================

df_filtered = df[df['year'] <= selected_year]
if selected_country != 'Все':
    df_filtered = df_filtered[df_filtered['country'] == selected_country]

# ============================================================
# ЗАГОЛОВОК
# ============================================================

st.title("🦠 VIGIL — Global Virus Surveillance")
st.markdown("**Virus Intelligence & Global Intervention Laboratory**")
st.caption("Мониторинг вируса Нипах (1998–2024) | Данные: WHO, CDC, открытые источники")

# ============================================================
# КАРТА МИРА
# ============================================================

st.subheader("🗺️ Карта вспышек")

# Создаём карту
m = folium.Map(location=[20, 80], zoom_start=3, tiles='CartoDB dark_matter')

for _, row in df_filtered.iterrows():
    radius = max(row['deaths'] * 2, 5)
    
    if row['cfr'] > 80:
        color = '#ff0000'
    elif row['cfr'] > 50:
        color = '#ff6600'
    else:
        color = '#ffcc00'
    
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']] if 'latitude' in df.columns else [0, 0],
        radius=radius,
        popup=f"{row['location']}<br>Deaths: {row['deaths']}<br>CFR: {row['cfr']}%",
        color=color,
        fill=True,
        fillColor=color,
        fillOpacity=0.7
    ).add_to(m)

folium_static(m, width=1200, height=500)

# ============================================================
# ГРАФИКИ
# ============================================================

col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Динамика случаев")
    fig_cases = px.line(
        df_filtered.groupby('year')['cases'].sum().reset_index(),
        x='year', y='cases',
        markers=True,
        color_discrete_sequence=['#00ccff'],
        labels={'year': 'Год', 'cases': 'Случаи'}
    )
    fig_cases.update_layout(template='plotly_dark', height=300)
    st.plotly_chart(fig_cases, use_container_width=True)

with col2:
    st.subheader("☠️ Летальность (CFR) по странам")
    cfr_by_country = df_filtered.groupby('country')['cfr'].mean().reset_index()
    fig_cfr = px.bar(
        cfr_by_country,
        x='country', y='cfr',
        color='cfr',
        color_continuous_scale='Reds',
        labels={'country': 'Страна', 'cfr': 'CFR (%)'}
    )
    fig_cfr.update_layout(template='plotly_dark', height=300)
    st.plotly_chart(fig_cfr, use_container_width=True)

# ============================================================
# ТАБЛИЦА ДАННЫХ
# ============================================================

st.subheader("📋 Данные по вспышкам")
st.dataframe(df_filtered[['year', 'location', 'country', 'cases', 'deaths', 'cfr']])

# ============================================================
# ПОДВАЛ
# ============================================================

st.markdown("---")
st.caption("🦠 VIGIL | Science for Future | Открытая наука для глобального здоровья")
