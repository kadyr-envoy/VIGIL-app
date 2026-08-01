from flask import Flask, render_template_string, request, Response, redirect
from flask_caching import Cache
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from datetime import datetime
import time
import os
from dotenv import load_dotenv

# Загружаем переменные окружения (если есть .env файл)
load_dotenv()

app = Flask(__name__)

app.config['CACHE_TYPE'] = 'SimpleCache'
app.config['CACHE_DEFAULT_TIMEOUT'] = 3600
cache = Cache(app)

MAX_DATA_POINTS = 500

@cache.cached(timeout=3600, key_prefix='data')
def load_data():
    # === ВСЕ ДАННЫЕ ВШИТЫ В КОД ===
    # Создаём пустой DataFrame
    df = pd.DataFrame()
    
    # Координаты для Nipah
    coords = {
        'Ipoh': [4.6, 101.1],
        'Kampung Sungai Nipah': [4.6, 101.1],
        'Meherpur': [23.8, 88.9],
        'Siliguri': [26.7, 88.4],
        'Naogaon': [24.9, 88.6],
        'Rajbari': [23.7, 89.5],
        'Nadia': [23.0, 88.5],
        'Manikganj': [23.9, 90.0],
        'Kozhikode': [11.3, 75.8],
        'Faridpur': [23.6, 89.8],
        'Dhaka': [23.8, 90.4],
        'Singapore': [1.3, 103.8]
    }
    
    # === NIPAH VIRUS ===
    nipah_data = [
        {'year': 1998, 'location': 'Ipoh', 'country': 'Malaysia', 'cases': 265, 'deaths': 105, 'cfr': 39.6},
        {'year': 1999, 'location': 'Singapore', 'country': 'Singapore', 'cases': 11, 'deaths': 1, 'cfr': 9.1},
        {'year': 2001, 'location': 'Meherpur', 'country': 'Bangladesh', 'cases': 13, 'deaths': 9, 'cfr': 69.2},
        {'year': 2001, 'location': 'Siliguri', 'country': 'India', 'cases': 66, 'deaths': 45, 'cfr': 68.2},
        {'year': 2003, 'location': 'Naogaon', 'country': 'Bangladesh', 'cases': 12, 'deaths': 8, 'cfr': 66.7},
        {'year': 2004, 'location': 'Rajbari', 'country': 'Bangladesh', 'cases': 36, 'deaths': 27, 'cfr': 75.0},
        {'year': 2007, 'location': 'Nadia', 'country': 'India', 'cases': 5, 'deaths': 5, 'cfr': 100.0},
        {'year': 2008, 'location': 'Manikganj', 'country': 'Bangladesh', 'cases': 20, 'deaths': 12, 'cfr': 60.0},
        {'year': 2018, 'location': 'Kozhikode', 'country': 'India', 'cases': 18, 'deaths': 17, 'cfr': 94.4},
        {'year': 2019, 'location': 'Faridpur', 'country': 'Bangladesh', 'cases': 8, 'deaths': 5, 'cfr': 62.5},
        {'year': 2021, 'location': 'Kozhikode', 'country': 'India', 'cases': 2, 'deaths': 2, 'cfr': 100.0},
        {'year': 2023, 'location': 'Dhaka', 'country': 'Bangladesh', 'cases': 14, 'deaths': 8, 'cfr': 57.1},
    ]
    nipah_df = pd.DataFrame(nipah_data)
    nipah_df['virus'] = 'Nipah'
    nipah_df['latitude'] = nipah_df['location'].map(lambda x: coords.get(x, [0,0])[0])
    nipah_df['longitude'] = nipah_df['location'].map(lambda x: coords.get(x, [0,0])[1])
    nipah_df['social_unrest'] = (nipah_df['cfr'] * 0.6 + nipah_df['deaths'] * 0.4).clip(0, 100).round(1)
    
    # === EBOLA VIRUS ===
    ebola_data = [
        {'year': 2014, 'location': 'Guinea', 'country': 'Guinea', 'cases': 3811, 'deaths': 2543, 'cfr': 66.7},
        {'year': 2014, 'location': 'Sierra Leone', 'country': 'Sierra Leone', 'cases': 14124, 'deaths': 3956, 'cfr': 28.0},
        {'year': 2014, 'location': 'Liberia', 'country': 'Liberia', 'cases': 10675, 'deaths': 4809, 'cfr': 45.0},
        {'year': 2018, 'location': 'DRC', 'country': 'DRC', 'cases': 54, 'deaths': 33, 'cfr': 61.1},
        {'year': 2022, 'location': 'Uganda', 'country': 'Uganda', 'cases': 164, 'deaths': 55, 'cfr': 33.5},
    ]
    ebola_df = pd.DataFrame(ebola_data)
    ebola_df['virus'] = 'Ebola'
    ebola_df['latitude'] = [9.5, 8.5, 6.5, -4.0, 1.3]
    ebola_df['longitude'] = [-13.5, -12.5, -9.5, 22.0, 32.4]
    ebola_df['social_unrest'] = (ebola_df['cfr'] * 0.6 + ebola_df['deaths'] * 0.4).clip(0, 100).round(1)
    
    # === MARBURG VIRUS ===
    marburg_data = [
        {'year': 2004, 'location': 'Uige', 'country': 'Angola', 'cases': 37, 'deaths': 34, 'cfr': 91.9},
        {'year': 2005, 'location': 'Uige', 'country': 'Angola', 'cases': 252, 'deaths': 227, 'cfr': 90.1},
        {'year': 2005, 'location': 'Luanda', 'country': 'Angola', 'cases': 56, 'deaths': 40, 'cfr': 71.4},
        {'year': 2007, 'location': 'Kamwenge', 'country': 'Uganda', 'cases': 4, 'deaths': 2, 'cfr': 50.0},
    ]
    marburg_df = pd.DataFrame(marburg_data)
    marburg_df['virus'] = 'Marburg'
    marburg_df['latitude'] = [-7.6, -7.6, -8.8, 0.2]
    marburg_df['longitude'] = [15.0, 15.0, 13.2, 30.4]
    marburg_df['social_unrest'] = (marburg_df['cfr'] * 0.6 + marburg_df['deaths'] * 0.4).clip(0, 100).round(1)
    
    # === HENDRA VIRUS ===
    hendra_data = [
        {'year': 1994, 'location': 'Brisbane', 'country': 'Australia', 'cases': 20, 'deaths': 4, 'cfr': 20.0},
        {'year': 1999, 'location': 'Cairns', 'country': 'Australia', 'cases': 15, 'deaths': 2, 'cfr': 13.3},
        {'year': 2004, 'location': 'Townsville', 'country': 'Australia', 'cases': 10, 'deaths': 1, 'cfr': 10.0},
        {'year': 2008, 'location': 'Brisbane', 'country': 'Australia', 'cases': 8, 'deaths': 1, 'cfr': 12.5},
        {'year': 2015, 'location': 'Cairns', 'country': 'Australia', 'cases': 12, 'deaths': 2, 'cfr': 16.7},
    ]
    hendra_df = pd.DataFrame(hendra_data)
    hendra_df['virus'] = 'Hendra'
    hendra_df['latitude'] = [-27.5, -16.9, -19.3, -27.5, -16.9]
    hendra_df['longitude'] = [153.0, 145.8, 146.8, 153.0, 145.8]
    hendra_df['social_unrest'] = (hendra_df['cfr'] * 0.6 + hendra_df['deaths'] * 0.4).clip(0, 100).round(1)
    
    # === DENGUE VIRUS ===
    dengue_data = [
        {'year': 2015, 'location': 'Manila', 'country': 'Philippines', 'cases': 100000, 'deaths': 200, 'cfr': 0.2},
        {'year': 2016, 'location': 'Rio de Janeiro', 'country': 'Brazil', 'cases': 150000, 'deaths': 300, 'cfr': 0.2},
        {'year': 2017, 'location': 'Delhi', 'country': 'India', 'cases': 80000, 'deaths': 150, 'cfr': 0.19},
        {'year': 2019, 'location': 'Dhaka', 'country': 'Bangladesh', 'cases': 100000, 'deaths': 200, 'cfr': 0.2},
        {'year': 2020, 'location': 'Bangkok', 'country': 'Thailand', 'cases': 60000, 'deaths': 100, 'cfr': 0.17},
        {'year': 2022, 'location': 'Singapore', 'country': 'Singapore', 'cases': 30000, 'deaths': 40, 'cfr': 0.13},
        {'year': 2023, 'location': 'Jakarta', 'country': 'Indonesia', 'cases': 40000, 'deaths': 80, 'cfr': 0.2},
    ]
    dengue_df = pd.DataFrame(dengue_data)
    dengue_df['virus'] = 'Dengue'
    dengue_df['latitude'] = [14.6, -22.9, 28.7, 23.8, 13.8, 1.3, -6.2]
    dengue_df['longitude'] = [120.9, -43.2, 77.2, 90.4, 100.5, 103.8, 106.8]
    dengue_df['social_unrest'] = (dengue_df['cfr'] * 0.6 + dengue_df['deaths'] * 0.4).clip(0, 100).round(1)
    
    # === WEST NILE VIRUS ===
    west_nile_data = [
        {'year': 2003, 'location': 'Colorado', 'country': 'USA', 'cases': 3000, 'deaths': 60, 'cfr': 2.0},
        {'year': 2012, 'location': 'Texas', 'country': 'USA', 'cases': 1900, 'deaths': 90, 'cfr': 4.7},
        {'year': 2018, 'location': 'California', 'country': 'USA', 'cases': 1500, 'deaths': 50, 'cfr': 3.3},
        {'year': 2020, 'location': 'Greece', 'country': 'Greece', 'cases': 400, 'deaths': 20, 'cfr': 5.0},
        {'year': 2022, 'location': 'Italy', 'country': 'Italy', 'cases': 600, 'deaths': 30, 'cfr': 5.0},
    ]
    west_nile_df = pd.DataFrame(west_nile_data)
    west_nile_df['virus'] = 'West Nile'
    west_nile_df['latitude'] = [39.0, 31.0, 36.8, 39.0, 41.9]
    west_nile_df['longitude'] = [-105.5, -99.0, -119.0, 21.0, 12.5]
    west_nile_df['social_unrest'] = (west_nile_df['cfr'] * 0.6 + west_nile_df['deaths'] * 0.4).clip(0, 100).round(1)
    
    # === CHIKUNGUNYA VIRUS ===
    chikungunya_data = [
        {'year': 2005, 'location': 'La Reunion', 'country': 'France', 'cases': 250000, 'deaths': 200, 'cfr': 0.08},
        {'year': 2013, 'location': 'Haiti', 'country': 'Haiti', 'cases': 50000, 'deaths': 30, 'cfr': 0.06},
        {'year': 2014, 'location': 'Colombia', 'country': 'Colombia', 'cases': 100000, 'deaths': 80, 'cfr': 0.08},
        {'year': 2017, 'location': 'Bangladesh', 'country': 'Bangladesh', 'cases': 10000, 'deaths': 10, 'cfr': 0.1},
        {'year': 2019, 'location': 'Brazil', 'country': 'Brazil', 'cases': 80000, 'deaths': 60, 'cfr': 0.075},
    ]
    chikungunya_df = pd.DataFrame(chikungunya_data)
    chikungunya_df['virus'] = 'Chikungunya'
    chikungunya_df['latitude'] = [-21.1, 19.0, 4.6, 23.8, -14.2]
    chikungunya_df['longitude'] = [55.5, -72.3, -74.1, 90.4, -51.9]
    chikungunya_df['social_unrest'] = (chikungunya_df['cfr'] * 0.6 + chikungunya_df['deaths'] * 0.4).clip(0, 100).round(1)
    
    # === ОБЪЕДИНЯЕМ ВСЕ ДАННЫЕ ===
    df_all = pd.concat([nipah_df, ebola_df, marburg_df, hendra_df, dengue_df, west_nile_df, chikungunya_df], ignore_index=True)
    
    if len(df_all) > MAX_DATA_POINTS:
        df_all = df_all.sample(MAX_DATA_POINTS)
    
    return df_all

def get_country_forecast(df, country):
    df_country = df[df['country'] == country]
    if len(df_country) < 3:
        return None, 0
    df_sorted = df_country.sort_values('year')
    X = df_sorted['year'].values.reshape(-1, 1)
    y = df_sorted['cases'].values
    if len(X) < 3:
        return None, 0
    model = LinearRegression()
    model.fit(X, y)
    future_years = np.arange(2025, 2030).reshape(-1, 1)
    predicted = model.predict(future_years)
    r2 = model.score(X, y)
    return predicted.tolist(), round(r2 * 100, 1)

@cache.cached(timeout=3600, key_prefix='charts')
def build_charts(virus='All', year='All', country='All'):
    start_time = time.time()
    df = load_data()
    
    if virus != 'All':
        df = df[df['virus'] == virus]
    if year != 'All':
        df = df[df['year'] == int(year)]
    if country != 'All':
        df = df[df['country'] == country]
    
    if df.empty:
        return empty_data()
    
    fig_map = px.scatter_geo(
        df,
        lat='latitude',
        lon='longitude',
        size='deaths',
        color='cfr',
        hover_name='location',
        title='GLOBAL OUTBREAKS',
        color_continuous_scale='Reds',
        size_max=40,
        projection='natural earth',
        labels={'cfr': 'CFR (%)', 'deaths': 'Fatalities'},
        hover_data={'cases': True, 'deaths': True, 'cfr': ':.1f%', 'virus': True},
        range_color=[0, 100]
    )
    fig_map.update_layout(
        template='plotly_dark',
        geo=dict(
            showland=True,
            landcolor='rgb(10, 10, 20)',
            coastlinecolor='rgb(30, 40, 60)',
            showocean=True,
            oceancolor='rgb(3, 3, 15)',
            showcountries=True,
            countrycolor='rgb(25, 35, 50)',
            showframe=False,
            bgcolor='rgba(0,0,0,0)'
        ),
        height=500,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#00ccff', family='Orbitron, monospace'),
        title_font=dict(size=16, color='#00ccff')
    )
    map_html = fig_map.to_html(full_html=False)
    
    # === УЛУЧШЕННАЯ АНИМИРОВАННАЯ КАРТА ===
    fig_anim = px.scatter_geo(
        df,
        lat='latitude',
        lon='longitude',
        size='deaths',
        color='cfr',
        hover_name='location',
        animation_frame='year',
        title='🌍 Virus Outbreaks (1994-2024)',
        color_continuous_scale='Reds',
        size_max=50,
        projection='natural earth',
        labels={'cfr': 'CFR (%)', 'deaths': 'Fatalities'},
        hover_data={
            'cases': True,
            'deaths': True,
            'cfr': ':.1f%',
            'virus': True,
            'country': True,
            'year': True
        },
        range_color=[0, 100],
        opacity=0.85
    )
    
    fig_anim.update_layout(
        template='plotly_dark',
        geo=dict(
            showland=True,
            landcolor='rgb(15, 15, 25)',
            coastlinecolor='rgb(40, 50, 70)',
            showocean=True,
            oceancolor='rgb(3, 3, 15)',
            showcountries=True,
            countrycolor='rgb(30, 40, 60)',
            showframe=False,
            bgcolor='rgba(0,0,0,0)',
            projection_scale=2.5,
            center=dict(lat=20, lon=60),
            lonaxis_range=[-20, 140],
            lataxis_range=[-30, 50]
        ),
        height=650,
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#00ccff', family='Orbitron, monospace', size=12),
        title_font=dict(size=20, color='#00ccff'),
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                dict(
                    label='▶ Play',
                    method='animate',
                    args=[None, {
                        'frame': {'duration': 500, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 300, 'easing': 'cubic-in-out'}
                    }]
                ),
                dict(
                    label='⏹ Stop',
                    method='animate',
                    args=[[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                ),
                dict(
                    label='⟳ Loop',
                    method='animate',
                    args=[None, {
                        'frame': {'duration': 500, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 300, 'easing': 'cubic-in-out'},
                        'loop': True
                    }]
                )
            ],
            'bgcolor': 'rgba(0,0,0,0.5)',
            'bordercolor': 'rgba(0,204,255,0.2)',
            'font': {'color': '#00ccff', 'size': 12}
        }],
        sliders=[{
            'currentvalue': {
                'prefix': '📅 ',
                'font': {'size': 16, 'color': '#00ccff'},
                'visible': True
            },
            'len': 0.9,
            'x': 0.05,
            'y': -0.15,
            'pad': {'t': 30},
            'bgcolor': 'rgba(0,0,0,0.3)',
            'bordercolor': 'rgba(0,204,255,0.1)',
            'font': {'color': '#8899aa'}
        }]
    )
    
    fig_anim.update_traces(
        marker=dict(
            line=dict(width=1.5, color='rgba(255,255,255,0.2)'),
            opacity=0.9
        ),
        selector=dict(mode='markers')
    )
    
    animation_map_html = fig_anim.to_html(full_html=False)
    
    # === ГРАФИК 1: ДИНАМИКА СЛУЧАЕВ ===
    fig1 = px.line(
        df.groupby('year')['cases'].sum().reset_index(),
        x='year',
        y='cases',
        markers=True,
        color_discrete_sequence=['#00ccff'],
        title='CASES OVER TIME'
    )
    fig1.update_layout(
        template='plotly_dark',
        height=250,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#00ccff', family='Orbitron', size=10),
        xaxis=dict(showgrid=False, color='#334455', tickfont=dict(size=9)),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,204,255,0.05)', tickfont=dict(size=9))
    )
    chart1 = fig1.to_html(full_html=False)
    
    # === ГРАФИК 2: CFR ПО СТРАНАМ ===
    fig2 = px.bar(
        df.groupby('country')['cfr'].mean().reset_index(),
        x='country',
        y='cfr',
        color='cfr',
        color_continuous_scale='Reds',
        title='CFR BY COUNTRY',
        labels={'cfr': 'CFR (%)', 'country': ''}
    )
    fig2.update_layout(
        template='plotly_dark',
        height=250,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ff4d4d', family='Orbitron', size=10),
        xaxis=dict(showgrid=False, color='#334455', tickfont=dict(size=9)),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,0,0,0.05)', title_font=dict(size=10)),
        coloraxis_colorbar=dict(
            title="CFR (%)",
            title_font=dict(color='#ff4d4d', size=9),
            tickfont=dict(color='#8899aa', size=8),
            len=0.5,
            thickness=12,
            x=1.02,
            y=0.5,
            ticks='outside',
            tickvals=[0, 25, 50, 75, 100],
            ticktext=['0%', '25%', '50%', '75%', '100%']
        )
    )
    fig2.update_traces(
        hovertemplate='<b>%{x}</b><br>CFR: %{y:.1f}%<extra></extra>'
    )
    chart2 = fig2.to_html(full_html=False)
    
    # === ГРАФИК 3: СОЦИАЛЬНОЕ НЕДОВОЛЬСТВО ===
    fig3 = px.bar(
        df.groupby('country')['social_unrest'].mean().reset_index(),
        x='country',
        y='social_unrest',
        color='social_unrest',
        color_continuous_scale='Reds',
        title='SOCIAL UNREST BY COUNTRY',
        labels={'social_unrest': 'Unrest Index (0-100)', 'country': ''}
    )
    fig3.update_layout(
        template='plotly_dark',
        height=250,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ff8800', family='Orbitron', size=10),
        xaxis=dict(showgrid=False, color='#334455', tickfont=dict(size=9)),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,136,0,0.05)', title_font=dict(size=10)),
        coloraxis_colorbar=dict(
            title="Unrest",
            title_font=dict(color='#ff8800', size=9),
            tickfont=dict(color='#8899aa', size=8),
            len=0.5,
            thickness=12,
            x=1.02,
            y=0.5,
            ticks='outside',
            tickvals=[0, 25, 50, 75, 100],
            ticktext=['0', '25', '50', '75', '100']
        )
    )
    fig3.update_traces(
        hovertemplate='<b>%{x}</b><br>Unrest Index: %{y:.1f}<extra></extra>'
    )
    chart3 = fig3.to_html(full_html=False)
    
    # === AI-ПРОГНОЗ ===
    df_sorted = df.sort_values('year')
    if len(df_sorted) >= 4:
        X = df_sorted['year'].values.reshape(-1, 1)
        y = df_sorted['cases'].values
        
        poly_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
        poly_model.fit(X, y)
        
        future_years = np.arange(2025, 2030).reshape(-1, 1)
        predicted = poly_model.predict(future_years)
        
        std_dev = np.std(y) if len(y) > 1 else 1
        lower_bound = predicted - std_dev * 0.5
        upper_bound = predicted + std_dev * 0.5
        r2 = poly_model.score(X, y) if len(X) > 2 else 0
        accuracy = round(r2 * 100, 1)
        
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            x=df_sorted['year'], y=df_sorted['cases'],
            mode='markers', name='Historical Data',
            marker=dict(color='#00ccff', size=8)
        ))
        fig_pred.add_trace(go.Scatter(
            x=future_years.flatten(), y=predicted,
            mode='lines+markers', name='AI Forecast',
            line=dict(color='#ff4d4d', dash='dash', width=3),
            marker=dict(color='#ff4d4d', size=10)
        ))
        fig_pred.add_trace(go.Scatter(
            x=np.concatenate([future_years.flatten(), future_years.flatten()[::-1]]),
            y=np.concatenate([upper_bound, lower_bound[::-1]]),
            fill='toself',
            fillcolor='rgba(255, 77, 77, 0.2)',
            line=dict(color='rgba(255, 77, 77, 0)'),
            name='Confidence Interval'
        ))
        fig_pred.update_layout(
            title='🤖 AI FORECAST (2025-2029)',
            template='plotly_dark',
            height=280,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#00ccff', family='Orbitron'),
            xaxis_title='YEAR',
            yaxis_title='CASES',
            xaxis=dict(showgrid=False, color='#334455'),
            yaxis=dict(showgrid=True, gridcolor='rgba(0,204,255,0.05)')
        )
        chart_pred = fig_pred.to_html(full_html=False)
    else:
        chart_pred = '<p style="color:#ff8800; text-align:center;">Not enough data for forecast</p>'
        accuracy = 0
    
    country_forecasts = {}
    for c in df['country'].unique():
        pred, acc = get_country_forecast(df, c)
        if pred is not None:
            country_forecasts[c] = {
                'v2025': int(pred[0]),
                'v2026': int(pred[1]),
                'v2027': int(pred[2]),
                'v2028': int(pred[3]),
                'v2029': int(pred[4]),
                'accuracy': acc
            }
    
    table_html = df[['year', 'location', 'country', 'cases', 'deaths', 'cfr', 'virus']].head(50).to_html(classes='table', index=False)
    
    # === ДНИ БЕЗ ВСПЫШКИ ===
    last_outbreak_year = df['year'].max()
    current_year = datetime.now().year
    days_since_last = (datetime(current_year, 12, 31) - datetime(last_outbreak_year, 12, 31)).days
    if days_since_last < 0:
        days_since_last = 0
    
    # === ИСТОРИЧЕСКИЙ МАКСИМУМ ===
    max_deaths_row = df.loc[df['deaths'].idxmax()]
    historical_max = {
        'location': max_deaths_row['location'],
        'year': int(max_deaths_row['year']),
        'deaths': int(max_deaths_row['deaths']),
        'virus': max_deaths_row['virus']
    }
    
    total_cases = int(df['cases'].sum())
    total_deaths = int(df['deaths'].sum())
    total_countries = df['country'].nunique()
    avg_cfr = round(df['cfr'].mean(), 1)
    virus_count = df['virus'].nunique()
    
    elapsed = round(time.time() - start_time, 2)
    
    # ============================================================
    # VIGIL: POLAR — КАРТА ЛЕДНИКОВ И ДРЕВНИХ ВИРУСОВ
    # ============================================================
    
    glaciers = [
        {'name': 'Greenland Ice Sheet', 'lat': 72.0, 'lon': -40.0, 'loss': 280, 'risk': 95},
        {'name': 'Antarctic Ice Sheet', 'lat': -75.0, 'lon': 60.0, 'loss': 150, 'risk': 88},
        {'name': 'Alaska Glacier', 'lat': 61.0, 'lon': -147.0, 'loss': 75, 'risk': 82},
        {'name': 'Himalayan Glacier', 'lat': 28.0, 'lon': 86.0, 'loss': 50, 'risk': 79},
        {'name': 'Siberian Permafrost', 'lat': 67.0, 'lon': 135.0, 'loss': 40, 'risk': 91},
        {'name': 'Patagonian Ice Field', 'lat': -50.0, 'lon': -73.0, 'loss': 25, 'risk': 70},
        {'name': 'Svalbard Ice Cap', 'lat': 78.0, 'lon': 20.0, 'loss': 30, 'risk': 76},
    ]
    
    ancient_viruses = [
        {'name': 'Pithovirus sibericum', 'lat': 67.5, 'lon': 134.0, 'year': 2014, 'status': 'Revived', 'risk': 92},
        {'name': 'Mollivirus sibericum', 'lat': 68.0, 'lon': 135.0, 'year': 2015, 'status': 'Revived', 'risk': 88},
        {'name': 'Pandoravirus', 'lat': 66.0, 'lon': 133.0, 'year': 2018, 'status': 'Identified', 'risk': 75},
        {'name': 'Megavirus chilensis', 'lat': -50.0, 'lon': -73.0, 'year': 2020, 'status': 'Identified', 'risk': 68},
        {'name': 'Cedratvirus', 'lat': 65.0, 'lon': 130.0, 'year': 2022, 'status': 'Identified', 'risk': 72},
    ]
    
    fig_polar = go.Figure()
    
    for g in glaciers:
        fig_polar.add_trace(go.Scattergeo(
            lon=[g['lon']],
            lat=[g['lat']],
            mode='markers',
            marker=dict(
                size=20 + g['loss'] * 0.1,
                color='#00ccff',
                opacity=0.85,
                symbol='circle',
                line=dict(width=1.5, color='rgba(255,255,255,0.3)')
            ),
            text=f"<b>{g['name']}</b><br>Ice loss: {g['loss']} Gt/year<br>Risk: {g['risk']}%",
            hoverinfo='text',
            name='Glaciers',
            showlegend=False
        ))
    
    for v in ancient_viruses:
        fig_polar.add_trace(go.Scattergeo(
            lon=[v['lon']],
            lat=[v['lat']],
            mode='markers',
            marker=dict(
                size=22,
                color='#ff4d4d',
                opacity=0.9,
                symbol='x-thin',
                line=dict(width=2, color='rgba(255,255,255,0.5)')
            ),
            text=f"<b>{v['name']}</b><br>Found: {v['year']}<br>Status: {v['status']}<br>Risk: {v['risk']}%",
            hoverinfo='text',
            name='Ancient Viruses',
            showlegend=False
        ))
    
    for g in glaciers:
        for v in ancient_viruses:
            if abs(g['lat'] - v['lat']) < 15 and abs(g['lon'] - v['lon']) < 20:
                fig_polar.add_trace(go.Scattergeo(
                    lon=[g['lon'], v['lon']],
                    lat=[g['lat'], v['lat']],
                    mode='lines',
                    line=dict(width=1.5, color='rgba(255, 77, 77, 0.25)', dash='dot'),
                    hoverinfo='none',
                    showlegend=False
                ))
    
    fig_polar.update_layout(
        template='plotly_dark',
        geo=dict(
            showland=True,
            landcolor='rgb(8, 8, 20)',
            coastlinecolor='rgb(30, 40, 60)',
            showocean=True,
            oceancolor='rgb(2, 2, 12)',
            showcountries=True,
            countrycolor='rgb(25, 35, 50)',
            showframe=False,
            bgcolor='rgba(0,0,0,0)',
            projection_scale=2.2,
            center=dict(lat=25, lon=50),
            lonaxis_range=[-40, 150],
            lataxis_range=[-40, 80]
        ),
        height=700,
        margin=dict(l=0, r=0, t=60, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(
            color='#00ccff',
            family='Orbitron, monospace',
            size=12
        ),
        title=dict(
            text='❄️ VIGIL: POLAR — ICE MELT & ANCIENT VIRUSES',
            font=dict(size=22, color='#00ccff'),
            x=0.5
        )
    )
    
    polar_map_html = fig_polar.to_html(full_html=False)
    
    return {
        'map_html': map_html,
        'animation_map_html': animation_map_html,
        'chart1': chart1,
        'chart2': chart2,
        'chart3': chart3,
        'chart_pred': chart_pred,
        'accuracy': accuracy,
        'country_forecasts': country_forecasts,
        'table_html': table_html,
        'total_cases': total_cases,
        'total_deaths': total_deaths,
        'total_countries': total_countries,
        'avg_cfr': avg_cfr,
        'virus_count': virus_count,
        'days_since_last': days_since_last,
        'historical_max': historical_max,
        'polar_map_html': polar_map_html,
        'last_updated': datetime.now().strftime("%d %B %Y, %H:%M UTC"),
        'load_time': elapsed,
        'viruses': sorted(load_data()['virus'].unique()),
        'years': sorted(load_data()['year'].unique()),
        'countries': ['All'] + sorted(load_data()['country'].unique()),
        'selected_virus': virus,
        'selected_year': year,
        'selected_country': country,
        'min_year': min(load_data()['year'].unique()),
        'max_year': max(load_data()['year'].unique())
    }

def empty_data():
    return {
        'map_html': '<p style="color:#ff4d4d; text-align:center;">No data for selected filters</p>',
        'animation_map_html': '<p style="color:#ff4d4d; text-align:center;">No data for animation</p>',
        'chart1': '<p style="color:#ff4d4d; text-align:center;">No data</p>',
        'chart2': '<p style="color:#ff4d4d; text-align:center;">No data</p>',
        'chart3': '<p style="color:#ff4d4d; text-align:center;">No data</p>',
        'chart_pred': '<p style="color:#ff4d4d; text-align:center;">Not enough data</p>',
        'accuracy': 0,
        'country_forecasts': {},
        'table_html': '<p style="color:#ff4d4d; text-align:center;">No data</p>',
        'total_cases': 0,
        'total_deaths': 0,
        'total_countries': 0,
        'avg_cfr': 0,
        'virus_count': 0,
        'days_since_last': 0,
        'historical_max': {'location': '—', 'year': '—', 'deaths': 0, 'virus': '—'},
        'polar_map_html': '<p style="color:#ff4d4d; text-align:center;">No data for POLAR map</p>',
        'last_updated': datetime.now().strftime("%d %B %Y, %H:%M UTC"),
        'load_time': 0,
        'viruses': [],
        'years': [],
        'countries': ['All'],
        'selected_virus': 'All',
        'selected_year': 'All',
        'selected_country': 'All',
        'min_year': 1998,
        'max_year': 2024
    }

@app.route('/', methods=['GET'])
def index():
    if request.args.get('refresh'):
        cache.delete('data')
        cache.delete('charts')
        return redirect('/')
    
    virus = request.args.get('virus', 'All')
    year = request.args.get('year', 'All')
    country = request.args.get('country', 'All')
    data = build_charts(virus, year, country)
    return render_template_string(HTML_TEMPLATE, **data)

@app.route('/download_csv')
def download_csv():
    df = load_data()
    csv = df.to_csv(index=False)
    return Response(csv, mimetype='text/csv', headers={'Content-Disposition': 'attachment;filename=vigil_data.csv'})

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>VIGIL · Global Virus Intelligence</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap" rel="stylesheet">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            background: #04040e; 
            color: #c8d0e0; 
            font-family: 'Orbitron', monospace; 
            min-height: 100vh; 
            overflow-x: hidden;
            transition: background 0.3s ease, color 0.3s ease;
            position: relative;
            padding-top: 0;
            margin: 0;
        }
        #particles-canvas {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 0;
            pointer-events: none;
        }
        body.light-mode {
            background: #f0f0f5;
            color: #1a1a2e;
        }
        body.light-mode .card-dark { background: rgba(255,255,255,0.9); border-color: rgba(0,0,0,0.1); }
        body.light-mode .stat-card { background: rgba(255,255,255,0.9); border-color: rgba(0,0,0,0.1); }
        body.light-mode .header { border-bottom-color: rgba(0,0,0,0.1); }
        body.light-mode .header .subtitle { color: #556677; }
        body.light-mode .badge-vigil { background: rgba(0,0,0,0.05); border-color: rgba(0,0,0,0.1); color: #0044ff; }
        body.light-mode .section-title { color: #0044ff; border-left-color: #0044ff; }
        body.light-mode .footer { color: #556677; border-top-color: rgba(0,0,0,0.1); }
        body.light-mode .stat-number { color: #0044ff; }
        body.light-mode .instruction { color: #556677; border-color: rgba(0,0,0,0.05); }
        body.light-mode .table { color: #1a1a2e; }
        body.light-mode .table th { color: #0044ff; }
        body.light-mode .last-updated { color: #556677; }
        body.light-mode .filter-bar { background: rgba(255,255,255,0.6); border-color: rgba(0,0,0,0.05); }
        body.light-mode .filter-bar select { background: white; color: #1a1a2e; border-color: rgba(0,0,0,0.1); }
        body.light-mode .nav-tab { color: #556677; }
        body.light-mode .nav-tab.active { color: #0044ff; border-bottom-color: #0044ff; }
        body.light-mode .about-card { background: rgba(255,255,255,0.9); border-color: rgba(0,0,0,0.1); }
        body.light-mode .about-card p { color: #556677; }
        body.light-mode .about-card h2 { color: #0044ff; }
        body.light-mode .theme-toggle { background: rgba(0,0,0,0.05); color: #1a1a2e; border-color: rgba(0,0,0,0.1); }
        body.light-mode .share-btn { background: rgba(0,0,0,0.05); color: #1a1a2e; border-color: rgba(0,0,0,0.1); }
        body.light-mode .filter-bar input[type="range"] { background: #ddd; }
        body.light-mode .filter-bar input[type="range"]::-webkit-slider-thumb { background: #0044ff; }
        body.light-mode .country-forecast { background: rgba(255,255,255,0.6); border-color: rgba(0,0,0,0.05); }
        
        body::before { content: ''; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: repeating-linear-gradient(0deg, rgba(0,204,255,0.015) 0px, rgba(0,204,255,0.015) 1px, transparent 1px, transparent 4px); pointer-events: none; z-index: 9998; animation: scan 8s linear infinite; }
        body.light-mode::before { display: none; }
        @keyframes scan { 0% { transform: translateY(-100%); } 100% { transform: translateY(100%); } }
        body::after { content: ''; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: radial-gradient(ellipse at 30% 50%, rgba(0,204,255,0.02) 0%, transparent 60%), radial-gradient(ellipse at 70% 50%, rgba(255,0,80,0.02) 0%, transparent 60%); pointer-events: none; z-index: 0; }
        body.light-mode::after { display: none; }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; position: relative; z-index: 1; }
        
        .header {
            text-align: center;
            padding: 25px 0 15px;
            border-bottom: 1px solid rgba(0,204,255,0.08);
            position: relative;
        }
        .header .logo {
            font-size: 4.2rem;
            font-weight: 900;
            background: linear-gradient(135deg, #00ccff, #0044ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: 14px;
            position: relative;
            transition: all 0.4s ease;
            filter: drop-shadow(0 0 30px rgba(0, 204, 255, 0.3))
                    drop-shadow(0 0 60px rgba(0, 204, 255, 0.1));
            animation: pulseGlow 3s ease-in-out infinite;
        }
        .header .logo::before {
            content: '';
            position: absolute;
            top: -50px;
            left: -50px;
            width: calc(100% + 100px);
            height: calc(100% + 100px);
            background: radial-gradient(ellipse at center, 
                        rgba(0, 204, 255, 0.12) 0%, 
                        rgba(0, 204, 255, 0.04) 30%, 
                        rgba(255, 0, 80, 0.02) 50%, 
                        transparent 70%);
            border-radius: 50%;
            pointer-events: none;
            z-index: -1;
            transition: all 0.6s ease;
            animation: auraPulse 4s ease-in-out infinite;
        }
        .header .logo:hover {
            filter: drop-shadow(0 0 60px rgba(0, 204, 255, 0.5))
                    drop-shadow(0 0 120px rgba(0, 204, 255, 0.2))
                    drop-shadow(0 0 200px rgba(255, 0, 80, 0.1));
            animation: none;
            transform: scale(1.02);
        }
        .header .logo:hover::before {
            background: radial-gradient(ellipse at center, 
                        rgba(0, 204, 255, 0.25) 0%, 
                        rgba(0, 204, 255, 0.08) 30%, 
                        rgba(255, 0, 80, 0.05) 50%, 
                        transparent 70%);
            transform: scale(1.2);
        }
        @keyframes pulseGlow {
            0%, 100% { filter: drop-shadow(0 0 30px rgba(0, 204, 255, 0.3)) drop-shadow(0 0 60px rgba(0, 204, 255, 0.1)); }
            50% { filter: drop-shadow(0 0 50px rgba(0, 204, 255, 0.5)) drop-shadow(0 0 100px rgba(0, 204, 255, 0.2)); }
        }
        @keyframes auraPulse {
            0%, 100% { opacity: 0.7; transform: scale(1); }
            50% { opacity: 1; transform: scale(1.05); }
        }
        
        .header .subtitle { font-size: 0.85rem; color: #445566; letter-spacing: 8px; font-weight: 300; margin-top: 2px; }
        .badge-vigil { display: inline-block; background: rgba(0,204,255,0.04); color: #00ccff; padding: 3px 14px; border-radius: 20px; font-size: 0.6rem; border: 1px solid rgba(0,204,255,0.06); letter-spacing: 2px; margin: 0 3px; transition: all 0.3s ease; }
        .badge-vigil:hover { background: rgba(0,204,255,0.08); border-color: rgba(0,204,255,0.15); box-shadow: 0 0 20px rgba(0,204,255,0.1); }
        
        .section-title { color: #00ccff; font-size: 1.3rem; margin-top: 40px; margin-bottom: 18px; padding-left: 16px; border-left: 2px solid #00ccff; letter-spacing: 4px; opacity: 0; transform: translateX(-15px); transition: all 0.7s ease; text-shadow: 0 0 10px rgba(0,204,255,0.3); }
        .section-title.visible { opacity: 1; transform: translateX(0); }
        .section-title:hover { text-shadow: 0 0 30px rgba(0,204,255,0.5), 0 0 80px rgba(255,0,80,0.15); border-left-color: #ff4d4d; }
        
        .share-btn {
            background: rgba(0,204,255,0.05);
            border: 1px solid rgba(0,204,255,0.1);
            color: #00ccff;
            padding: 6px 14px;
            border-radius: 20px;
            font-family: 'Orbitron', monospace;
            font-size: 0.6rem;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .share-btn:hover { background: rgba(0,204,255,0.15); box-shadow: 0 0 20px rgba(0,204,255,0.1); }
        .theme-toggle {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            color: #c8d0e0;
            padding: 6px 14px;
            border-radius: 20px;
            font-family: 'Orbitron', monospace;
            font-size: 0.6rem;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .theme-toggle:hover { background: rgba(255,255,255,0.1); }
        .music-btn {
            background: rgba(0,204,255,0.05);
            border: 1px solid rgba(0,204,255,0.1);
            color: #00ccff;
            padding: 6px 14px;
            border-radius: 20px;
            font-family: 'Orbitron', monospace;
            font-size: 0.6rem;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .music-btn:hover { background: rgba(0,204,255,0.15); box-shadow: 0 0 20px rgba(0,204,255,0.1); }
        .refresh-btn {
            background: rgba(0,204,255,0.05);
            border: 1px solid rgba(0,204,255,0.1);
            color: #00ccff;
            padding: 6px 14px;
            border-radius: 20px;
            font-family: 'Orbitron', monospace;
            font-size: 0.6rem;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .refresh-btn:hover { background: rgba(0,204,255,0.15); box-shadow: 0 0 20px rgba(0,204,255,0.1); }
        
        .filter-bar { display: flex; flex-wrap: wrap; gap: 15px; align-items: center; margin-bottom: 20px; padding: 15px; background: rgba(6,6,18,0.6); border-radius: 12px; border: 1px solid rgba(0,204,255,0.05); }
        .filter-bar select { background: rgba(6,6,18,0.9); color: #c8d0e0; border: 1px solid rgba(0,204,255,0.1); padding: 8px 15px; border-radius: 8px; font-family: 'Orbitron', monospace; font-size: 0.7rem; }
        .filter-bar select:focus { outline: none; border-color: #00ccff; }
        .filter-bar label { font-size: 0.6rem; color: #445566; margin-right: 5px; }
        .filter-bar input[type="range"] { width: 150px; height: 4px; -webkit-appearance: none; background: #1a2a3a; border-radius: 10px; outline: none; }
        .filter-bar input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; width: 14px; height: 14px; border-radius: 50%; background: #00ccff; cursor: pointer; }
        .filter-bar .year-display { font-size: 0.7rem; color: #00ccff; min-width: 40px; text-align: center; }
        .btn-download { background: rgba(0,204,255,0.1); color: #00ccff; border: 1px solid rgba(0,204,255,0.2); padding: 8px 20px; border-radius: 8px; font-family: 'Orbitron', monospace; font-size: 0.7rem; text-decoration: none; transition: all 0.3s ease; cursor: pointer; }
        .btn-download:hover { background: rgba(0,204,255,0.2); box-shadow: 0 0 20px rgba(0,204,255,0.1); }
        .stats-row { margin: 20px 0 30px; }
        .stat-card { background: rgba(6, 6, 18, 0.85); border: 1px solid rgba(0,204,255,0.08); border-radius: 12px; padding: 18px 10px; text-align: center; backdrop-filter: blur(8px); transition: all 0.3s ease; opacity: 0; transform: translateY(20px); transition: opacity 0.7s ease, transform 0.7s ease, border-color 0.3s ease, box-shadow 0.3s ease; }
        .stat-card.visible { opacity: 1; transform: translateY(0); }
        .stat-card:hover { border-color: rgba(0,204,255,0.2); box-shadow: 0 0 30px rgba(0,204,255,0.1), 0 0 60px rgba(255,0,80,0.05); transform: translateY(-2px); }
        .stat-number { display: block; font-size: 2.2rem; font-weight: 700; color: #00ccff; letter-spacing: 2px; font-family: 'Orbitron', monospace; }
        .stat-label { display: block; font-size: 0.6rem; color: #445566; letter-spacing: 3px; margin-top: 4px; }
        .card-dark { background: rgba(6, 6, 18, 0.9); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px; border: 1px solid rgba(0,204,255,0.04); box-shadow: 0 4px 24px rgba(0,0,0,0.5); transition: all 0.5s ease; opacity: 0; transform: translateY(25px); transition: opacity 0.8s cubic-bezier(0.22, 0.61, 0.36, 1), transform 0.8s cubic-bezier(0.22, 0.61, 0.36, 1), border-color 0.3s ease, box-shadow 0.3s ease; }
        .card-dark.visible { opacity: 1; transform: translateY(0); }
        .card-dark:hover { border-color: rgba(0,204,255,0.15); box-shadow: 0 0 40px rgba(0,204,255,0.05), 0 0 80px rgba(255,0,80,0.02); }
        .table { color: #b0c0d0; background: transparent; font-family: 'Orbitron', monospace; font-size: 0.7rem; }
        .table th { color: #00ccff; border-bottom: 1px solid rgba(0,204,255,0.06); font-weight: 400; letter-spacing: 2px; padding: 10px; text-transform: uppercase; }
        .table td { padding: 8px 10px; border-color: rgba(255,255,255,0.02); }
        .table tr:hover { background: rgba(0,204,255,0.02); }
        .instruction { background: rgba(0,204,255,0.02); padding: 8px 16px; border-radius: 8px; font-size: 0.7rem; color: #556677; margin-bottom: 12px; border: 1px solid rgba(0,204,255,0.03); letter-spacing: 2px; }
        .instruction strong { color: #00ccff; }
        .footer { text-align: center; padding: 25px 0 12px; color: #223344; font-size: 0.7rem; border-top: 1px solid rgba(0,204,255,0.03); margin-top: 40px; letter-spacing: 4px; opacity: 0; transform: translateY(15px); transition: all 1s ease; }
        .footer.visible { opacity: 1; transform: translateY(0); }
        .last-updated { font-size: 0.6rem; color: #334455; letter-spacing: 2px; margin-top: 5px; }
        .load-time { font-size: 0.6rem; color: #334455; margin-top: 5px; }
        .nav-tabs { display: flex; gap: 0; margin: 20px 0 30px; border-bottom: 1px solid rgba(0,204,255,0.1); }
        .nav-tab { padding: 12px 25px; font-family: 'Orbitron', monospace; font-size: 0.7rem; color: #556677; border: none; background: transparent; cursor: pointer; transition: all 0.3s ease; border-bottom: 2px solid transparent; letter-spacing: 2px; }
        .nav-tab:hover { color: #00ccff; }
        .nav-tab.active { color: #00ccff; border-bottom-color: #00ccff; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .about-card { background: rgba(6, 6, 18, 0.9); border-radius: 16px; padding: 35px; border: 1px solid rgba(0,204,255,0.08); margin-top: 20px; }
        .about-card h2 { color: #00ccff; font-size: 1.8rem; }
        .about-card p { color: #8899aa; font-size: 0.95rem; line-height: 1.8; }
        .about-card .highlight { color: #00ccff; }
        .country-forecast { background: rgba(6,6,18,0.5); border-radius: 12px; padding: 15px; margin-bottom: 10px; border: 1px solid rgba(0,204,255,0.05); transition: all 0.3s ease; }
        .country-forecast:hover { border-color: rgba(0,204,255,0.15); background: rgba(6,6,18,0.7); }
        .country-forecast h5 { color: #00ccff; font-size: 0.9rem; margin-bottom: 5px; }
        .country-forecast .pred-value { color: #ff4d4d; font-weight: 700; }
        .methodology-section { margin-top: 20px; }
        .methodology-section h4 { color: #00ccff; font-size: 1.1rem; margin-top: 20px; }
        .methodology-section ul { color: #8899aa; font-size: 0.85rem; line-height: 2; }
        .methodology-section code { color: #ff8800; background: rgba(255,136,0,0.1); padding: 2px 6px; border-radius: 4px; }
        #topBtn {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: rgba(0,204,255,0.1);
            border: 1px solid rgba(0,204,255,0.2);
            color: #00ccff;
            padding: 12px 16px;
            border-radius: 50%;
            font-size: 1.2rem;
            cursor: pointer;
            display: none;
            transition: all 0.3s ease;
            z-index: 9999;
        }
        #topBtn:hover { background: rgba(0,204,255,0.2); box-shadow: 0 0 20px rgba(0,204,255,0.1); }
        
        .typewriter {
            overflow: hidden;
            white-space: nowrap;
            margin: 0 auto;
            animation: typing 2s steps(40) 0.5s 1 normal both;
        }
        @keyframes typing {
            from { width: 0; }
            to { width: 100%; }
        }
        
        .polar-title {
            font-size: 1.8rem;
            font-weight: 700;
            background: linear-gradient(90deg, #00ccff, #ff4d4d, #00ccff);
            background-size: 300% 100%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradientShift 4s ease-in-out infinite;
            display: inline-block;
        }
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        @media (max-width: 768px) { .header .logo { font-size: 2.5rem; letter-spacing: 6px; } .section-title { font-size: 1rem; } .container { padding: 10px; } .stat-number { font-size: 1.6rem; } .filter-bar { flex-direction: column; align-items: stretch; } }
    </style>
</head>
<body>
    <canvas id="particles-canvas"></canvas>
    
    <div class="container">
        <div class="header">
            <div class="logo typewriter">VIGIL</div>
            <div class="subtitle">VIRUS INTELLIGENCE · GLOBAL INTERVENTION LABORATORY</div>
            <div style="margin-top: 12px;">
                <span class="badge-vigil">⚡ EST. 2026</span>
                <span class="badge-vigil">🔬 OPEN SCIENCE</span>
                <span class="badge-vigil">🤖 AI-POWERED</span>
                <span class="badge-vigil">🌍 GLOBAL HEALTH</span>
                <span class="badge-vigil">🦠 {{ virus_count }} VIRUSES</span>
                <button class="theme-toggle" onclick="toggleTheme()">🌓 THEME</button>
                <button class="share-btn" onclick="sharePage()">📤 SHARE</button>
                <button class="refresh-btn" onclick="refreshData()">🔄 REFRESH</button>
                <button id="musicBtn" class="music-btn">🔊 MUSIC</button>
            </div>
            <div class="last-updated">🔄 Last updated: {{ last_updated }}</div>
            <div class="load-time">⚡ Load time: {{ load_time }}s</div>
        </div>

        <div class="nav-tabs">
            <button class="nav-tab active" onclick="switchTab('dashboard')">📊 DASHBOARD</button>
            <button class="nav-tab" onclick="switchTab('forecast')">📈 COUNTRY FORECAST</button>
            <button class="nav-tab" onclick="switchTab('animation')">🎬 ANIMATION</button>
            <button class="nav-tab" onclick="switchTab('polar')">❄️ POLAR</button>
            <button class="nav-tab" onclick="switchTab('methodology')">📖 METHODOLOGY</button>
            <button class="nav-tab" onclick="switchTab('about')">ℹ️ ABOUT</button>
        </div>

        <div id="tab-dashboard" class="tab-content active">
            <div class="filter-bar">
                <div><label>🦠 VIRUS</label><select name="virus" onchange="applyFilters()"><option value="All">All</option>{% for v in viruses %}<option value="{{ v }}" {% if v == selected_virus %}selected{% endif %}>{{ v }}</option>{% endfor %}</select></div>
                <div>
                    <label>📅 YEAR</label>
                    <input type="range" id="yearSlider" min="{{ min_year }}" max="{{ max_year }}" value="{{ selected_year if selected_year != 'All' else max_year }}" step="1" oninput="updateYear(this.value)">
                    <span class="year-display" id="yearDisplay">{{ selected_year if selected_year != 'All' else max_year }}</span>
                </div>
                <div><label>🌍 COUNTRY</label><select name="country" onchange="applyFilters()">{% for c in countries %}<option value="{{ c }}" {% if c == selected_country %}selected{% endif %}>{{ c }}</option>{% endfor %}</select></div>
                <a href="/download_csv" class="btn-download" style="margin-left:auto;">⬇ DOWNLOAD CSV</a>
            </div>

            <div class="row stats-row">
                <div class="col-md-3"><div class="stat-card stat-item glow-hover"><span class="stat-number">{{ total_cases }}</span><span class="stat-label">TOTAL CASES</span></div></div>
                <div class="col-md-3"><div class="stat-card stat-item glow-hover"><span class="stat-number">{{ total_deaths }}</span><span class="stat-label">TOTAL DEATHS</span></div></div>
                <div class="col-md-3"><div class="stat-card stat-item glow-hover"><span class="stat-number">{{ total_countries }}</span><span class="stat-label">AFFECTED COUNTRIES</span></div></div>
                <div class="col-md-3"><div class="stat-card stat-item glow-hover"><span class="stat-number">{{ avg_cfr }}%</span><span class="stat-label">AVG CFR</span></div></div>
            </div>
            <div class="row stats-row">
                <div class="col-md-6"><div class="stat-card stat-item glow-hover"><span class="stat-number">{{ days_since_last }}</span><span class="stat-label">DAYS SINCE LAST OUTBREAK</span></div></div>
                <div class="col-md-6"><div class="stat-card stat-item glow-hover"><span class="stat-number">{{ historical_max.deaths }}</span><span class="stat-label">MOST DEATHS — {{ historical_max.location }} ({{ historical_max.year }})</span></div></div>
            </div>

            <div class="section-title scroll-title">🗺️ GLOBAL OUTBREAK MAP</div>
            <div class="instruction">▶ <strong>HOVER</strong> or <strong>CLICK</strong> on points for details</div>
            <div class="card-dark glow-hover" id="mapCard">{{ map_html|safe }}</div>

            <div class="section-title scroll-title">🤖 AI FORECAST</div>
            <div class="instruction">🔮 Forecast using polynomial regression (degree 2)</div>
            <div class="card-dark glow-hover scroll-card">{{ chart_pred|safe }}</div>
            <div style="text-align:center; margin-top:10px; color:#445566; font-size:0.8rem;">Model Accuracy: <span style="color:#00ccff;">{{ accuracy }}%</span></div>

            <div class="section-title scroll-title">📊 ANALYTICS</div>
            <div class="row">
                <div class="col-md-4"><div class="card-dark glow-hover scroll-card">{{ chart1|safe }}</div></div>
                <div class="col-md-4"><div class="card-dark glow-hover scroll-card">{{ chart2|safe }}</div></div>
                <div class="col-md-4"><div class="card-dark glow-hover scroll-card">{{ chart3|safe }}</div></div>
            </div>

            <div class="section-title scroll-title">📋 OUTBREAK DATA</div>
            <div class="card-dark glow-hover scroll-card" style="overflow-x:auto;">{{ table_html|safe }}</div>
        </div>

        <div id="tab-forecast" class="tab-content">
            <div class="section-title" style="opacity:1; transform:none;">📈 COUNTRY FORECASTS (2025-2029)</div>
            <div class="instruction">📊 AI predictions for each country based on historical data</div>
            <div class="row">
                {% for country, data in country_forecasts.items() %}
                <div class="col-md-4">
                    <div class="country-forecast">
                        <h5>{{ country }}</h5>
                        <div style="font-size:0.7rem; color:#445566;">
                            Accuracy: <span style="color:#00ccff;">{{ data.accuracy }}%</span>
                        </div>
                        <div style="font-size:0.65rem; color:#8899aa; margin-top:5px;">
                            2025: <span class="pred-value">{{ data.v2025 }}</span> cases<br>
                            2026: <span class="pred-value">{{ data.v2026 }}</span> cases<br>
                            2027: <span class="pred-value">{{ data.v2027 }}</span> cases<br>
                            2028: <span class="pred-value">{{ data.v2028 }}</span> cases<br>
                            2029: <span class="pred-value">{{ data.v2029 }}</span> cases
                        </div>
                    </div>
                </div>
                {% endfor %}
                {% if country_forecasts|length == 0 %}
                <div class="col-12"><p style="color:#ff8800; text-align:center;">No data for country forecasts</p></div>
                {% endif %}
            </div>
        </div>

        <div id="tab-animation" class="tab-content">
            <div class="section-title" style="opacity:1; transform:none;">🎬 VIRUS OUTBREAK ANIMATION</div>
            <div class="instruction">
                ▶️ <strong>PLAY</strong> — watch outbreaks appear year by year<br>
                🎯 <strong>HOVER</strong> — see detailed information about each outbreak<br>
                📅 <strong>SLIDER</strong> — jump to any year manually<br>
                🔄 <strong>LOOP</strong> — repeat the animation
            </div>
            <div class="card-dark" style="opacity:1; transform:none; padding: 10px;">
                {{ animation_map_html|safe }}
            </div>
            <div style="text-align:center; margin-top:10px; color:#445566; font-size:0.7rem;">
                🟢 <span style="color:#00ccff;">Size</span> = number of deaths &nbsp;·&nbsp; 
                🔴 <span style="color:#ff4d4d;">Color</span> = CFR (Case Fatality Rate)
            </div>
        </div>

        <div id="tab-polar" class="tab-content">
            <div class="section-title" style="opacity:1; transform:none;">
                <span class="polar-title">❄️ VIGIL: POLAR</span>
            </div>
            <div class="instruction">
                🔵 <strong>Glaciers</strong> — ice loss in real time (size = loss rate)<br>
                🔴 <strong>Ancient Viruses</strong> — pathogens revived from permafrost<br>
                ⚪ <strong>Lines</strong> — show connection between melting ice and emerging viral threats
            </div>
            <div class="card-dark" style="opacity:1; transform:none; padding: 10px;">
                {{ polar_map_html|safe }}
            </div>
            <div style="text-align:center; margin-top:10px; color:#445566; font-size:0.7rem;">
                ❄️ <span style="color:#00ccff;">Ice loss</span> measured in Gigatonnes/year &nbsp;·&nbsp; 
                🧬 <span style="color:#ff4d4d;">Ancient viruses</span> found in permafrost
            </div>
        </div>

        <div id="tab-methodology" class="tab-content">
            <div class="section-title" style="opacity:1; transform:none;">📖 METHODOLOGY</div>
            <div class="about-card methodology-section">
                <h2>How VIGIL Works</h2>
                <p><strong class="highlight">Data Sources</strong><br>
                All data is collected from open sources: WHO, CDC, GISAID, and peer-reviewed scientific publications.</p>
                
                <h4>🤖 AI Forecasting Model</h4>
                <p>We use <strong class="highlight">Polynomial Regression (degree 2)</strong> to predict future outbreaks. This model captures non-linear trends in historical data, providing more accurate forecasts than simple linear models.</p>
                <ul>
                    <li><strong>Training data:</strong> Historical outbreak data (cases, deaths, CFR)</li>
                    <li><strong>Prediction horizon:</strong> 5 years (2025-2029)</li>
                    <li><strong>Confidence interval:</strong> ±0.5 standard deviation</li>
                    <li><strong>Accuracy metric:</strong> R² score</li>
                </ul>
                
                <h4>📊 Social Unrest Index</h4>
                <p>This is a composite metric calculated as:<br>
                <code>Social Unrest = (CFR × 0.6) + (Deaths × 0.4)</code><br>
                It indicates the potential for public dissatisfaction during outbreaks.</p>
                
                <h4>🌍 Geographic Data</h4>
                <p>Coordinates are manually curated for accuracy. Future versions will use automated geocoding.</p>
                
                <h4>🔬 Limitations</h4>
                <ul>
                    <li>Data availability varies by country</li>
                    <li>Model accuracy depends on data quantity</li>
                    <li>Forecasts are for research purposes only</li>
                </ul>
                <div style="margin-top:20px; padding:15px; background:rgba(0,204,255,0.02); border-radius:8px; border-left:3px solid #00ccff;">
                    <p style="color:#556677; font-size:0.75rem; letter-spacing:1px;">
                        🔬 <em>"Science does not wait. We do not wait. The future starts now."</em>
                    </p>
                </div>
            </div>
        </div>

        <div id="tab-about" class="tab-content">
            <div class="section-title" style="opacity:1; transform:none;">ℹ️ ABOUT VIGIL</div>
            <div class="about-card">
                <h2>Virus Intelligence & Global Intervention Laboratory</h2>
                <p><strong class="highlight">VIGIL</strong> is an open-source global surveillance system for emerging infectious diseases. Built by <strong class="highlight">Abdylkadr Magomedov</strong>, a 16-year-old independent researcher from Russia.</p>
                <p>Our mission: to provide real-time intelligence on viral outbreaks, predict future threats using AI, and make scientific data accessible to everyone — for free.</p>
                <p><strong class="highlight">Science for Future</strong> is the movement behind VIGIL. We believe that knowledge belongs to all of humanity.</p>
                <div style="margin-top:20px; display:flex; gap:12px; flex-wrap:wrap;"><span class="badge-vigil">🌍 GLOBAL HEALTH</span><span class="badge-vigil">🧬 AI-POWERED</span><span class="badge-vigil">🔓 OPEN SCIENCE</span><span class="badge-vigil">🦠 VIRUS INTELLIGENCE</span></div>
                <div style="margin-top:20px; padding:15px; background:rgba(0,204,255,0.02); border-radius:8px; border-left:3px solid #00ccff;"><p style="color:#556677; font-size:0.75rem; letter-spacing:1px;">🔬 <em>"Science does not wait. We do not wait. The future starts now."</em></p></div>
                <div style="margin-top:15px; display:flex; gap:15px; flex-wrap:wrap;"><a href="https://kadyr-envoy.github.io/science-for-future/" target="_blank" class="btn-download">🌐 Science for Future</a><a href="https://github.com/kadyr-envoy" target="_blank" class="btn-download">🐙 GitHub</a><a href="https://t.me/science_for_future_official" target="_blank" class="btn-download">📱 Telegram</a></div>
            </div>
        </div>

        <div class="footer scroll-footer">
            <p>🦠 VIGIL · SCIENCE FOR FUTURE · OPEN SCIENCE FOR GLOBAL HEALTH</p>
            <p style="font-size: 0.6rem; color: #1a2a3a;">
                DATA: WHO · CDC · OPEN SOURCES · BUILT BY ABDYLKADR MAGOMEDOV · 2026
                &nbsp;·&nbsp; <a href="https://github.com/kadyr-envoy" target="_blank" style="color:#00ccff;text-decoration:none;">🐙 GitHub</a>
            </p>
        </div>
    </div>
    <button onclick="scrollToTop()" id="topBtn">⬆</button>
    
    <audio id="bgMusic" loop preload="auto">
        <source src="/static/track.mp3" type="audio/mpeg">
    </audio>
    
    <script>
        const audio = document.getElementById('bgMusic');
        const btn = document.getElementById('musicBtn');
        let isPlaying = false;

        btn.addEventListener('click', function() {
            if (isPlaying) {
                audio.pause();
                isPlaying = false;
                btn.textContent = '🔇 MUSIC';
            } else {
                audio.play().catch(() => {
                    alert('Click the page first, then try again.');
                });
                isPlaying = true;
                btn.textContent = '🔊 MUSIC';
            }
        });

        function refreshData() {
            const btn = document.querySelector('.refresh-btn');
            btn.textContent = '⏳ LOADING...';
            btn.disabled = true;
            const url = new URL(window.location.href);
            url.searchParams.set('refresh', Date.now());
            window.location.href = url.toString();
        }

        const canvas = document.getElementById('particles-canvas');
        const ctx = canvas.getContext('2d');
        let particles = [];
        let mouseX = 0;
        let mouseY = 0;

        function resizeCanvas() {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        }
        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);

        class Particle {
            constructor() {
                this.x = Math.random() * canvas.width;
                this.y = Math.random() * canvas.height;
                this.size = Math.random() * 2 + 0.5;
                this.speedX = (Math.random() - 0.5) * 0.5;
                this.speedY = (Math.random() - 0.5) * 0.5;
                this.opacity = Math.random() * 0.5 + 0.2;
            }
            update() {
                this.x += this.speedX;
                this.y += this.speedY;
                if (this.x < 0 || this.x > canvas.width) this.speedX *= -1;
                if (this.y < 0 || this.y > canvas.height) this.speedY *= -1;
                const dx = mouseX - this.x;
                const dy = mouseY - this.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 200) {
                    this.x += dx / dist * 0.2;
                    this.y += dy / dist * 0.2;
                }
            }
            draw() {
                ctx.beginPath();
                ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
                const isLight = document.body.classList.contains('light-mode');
                const color = isLight ? '0, 68, 255' : '0, 204, 255';
                ctx.fillStyle = `rgba(${color}, ${this.opacity})`;
                ctx.fill();
            }
        }

        function initParticles() {
            particles = [];
            for (let i = 0; i < 80; i++) {
                particles.push(new Particle());
            }
        }
        initParticles();

        function animateParticles() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            particles.forEach(p => {
                p.update();
                p.draw();
            });
            requestAnimationFrame(animateParticles);
        }
        animateParticles();

        document.addEventListener('mousemove', (e) => {
            mouseX = e.clientX;
            mouseY = e.clientY;
        });

        function updateYear(val) {
            document.getElementById('yearDisplay').textContent = val;
            clearTimeout(window.yearTimeout);
            window.yearTimeout = setTimeout(() => {
                applyFilters();
            }, 300);
        }

        function applyFilters() {
            const virus = document.querySelector('select[name="virus"]').value;
            const year = document.getElementById('yearSlider').value;
            const country = document.querySelector('select[name="country"]').value;
            window.location.href = '?virus=' + virus + '&year=' + year + '&country=' + country;
        }

        function switchTab(tab) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.nav-tab').forEach(el => el.classList.remove('active'));
            document.getElementById('tab-' + tab).classList.add('active');
            event.target.classList.add('active');
        }

        function toggleTheme() {
            document.body.classList.toggle('light-mode');
            localStorage.setItem('vigil-theme', document.body.classList.contains('light-mode') ? 'light' : 'dark');
        }

        function sharePage() {
            const url = "https://vigil-35oz.onrender.com";
            if (navigator.share) {
                navigator.share({ title: 'VIGIL - Global Virus Intelligence', url: url });
            } else {
                navigator.clipboard.writeText(url).then(() => {
                    alert('Link copied to clipboard!');
                }).catch(() => {
                    prompt('Copy this link:', url);
                });
            }
        }

        function scrollToTop() {
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }

        window.onscroll = function() {
            const btn = document.getElementById('topBtn');
            if (document.body.scrollTop > 300 || document.documentElement.scrollTop > 300) {
                btn.style.display = 'block';
            } else {
                btn.style.display = 'none';
            }
        };

        if (localStorage.getItem('vigil-theme') === 'light') {
            document.body.classList.add('light-mode');
        }

        document.addEventListener('DOMContentLoaded', function() {
            const urlParams = new URLSearchParams(window.location.search);
            const yearParam = urlParams.get('year');
            if (yearParam && yearParam !== 'All') {
                const slider = document.getElementById('yearSlider');
                if (slider) slider.value = yearParam;
                const display = document.getElementById('yearDisplay');
                if (display) display.textContent = yearParam;
            }

            const statItems = document.querySelectorAll('.stat-item');
            const statObserver = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) { entry.target.classList.add('visible'); }
                    else { entry.target.classList.remove('visible'); }
                });
            }, { threshold: 0.2 });
            statItems.forEach(item => statObserver.observe(item));

            const cards = document.querySelectorAll('.scroll-card');
            const titles = document.querySelectorAll('.scroll-title');
            const footer = document.querySelector('.scroll-footer');
            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) { entry.target.classList.add('visible'); }
                    else { entry.target.classList.remove('visible'); }
                });
            }, { threshold: 0.1, rootMargin: '0px 0px -20px 0px' });
            cards.forEach(card => observer.observe(card));
            titles.forEach(title => observer.observe(title));
            if (footer) {
                const fo = new IntersectionObserver((e) => {
                    e.forEach(entry => {
                        if (entry.isIntersecting) footer.classList.add('visible');
                        else footer.classList.remove('visible');
                    });
                }, { threshold: 0.1 });
                fo.observe(footer);
            }
            const mapCard = document.getElementById('mapCard');
            if (mapCard) {
                const mapObserver = new IntersectionObserver((e) => {
                    e.forEach(entry => {
                        if (entry.isIntersecting) mapCard.classList.add('visible');
                        else mapCard.classList.remove('visible');
                    });
                }, { threshold: 0.1 });
                mapObserver.observe(mapCard);
                setTimeout(() => mapCard.classList.add('visible'), 200);
            }
        });
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    app.run()
