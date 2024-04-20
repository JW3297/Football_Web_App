# General
import pandas as pd
import numpy as np
import io

# Plotting
from mplsoccer import Radar, grid
import matplotlib.colors as mcolors
from PIL import Image
import urllib

# Dashboard
import streamlit as st

home_colors = {
    'Arsenal': '#E63636',
    'Aston Villa' : '#811331',
    'Brentford' : '#E63636', 
    'Bournemouth' : '#E63636',
    'Brighton': '#2d8ce7',
    'Chelsea' : '#2D5DE7',
    'Crystal Palace' : '#E63636',
    'Everton' : '#2D5DE7', 
    'Fulham' : '#9C9C9C', 
    'Leeds' : '#9C9C9C', 
    'Leicester' : '#2d8ce7',
    'Liverpool' : '#E63636',
    'Nottingham Forest' : '#E63636',
    'Newcastle' : '#9C9C9C',
    'Man Utd' : '#E63636',
    'Man City' : '#6FC6E6',
    'Southampton' : '#E63636',
    'Tottenham' : '#9C9C9C', 
    'West Ham' : '#811331',
    'Wolves' : '#DA9D0A',    
    "Burnley": "#811331", 
    "Sheff Utd": "#E63636", 
    "Luton": "#FB6B07"
}

logos = {
    'Arsenal': 9825,
    'Aston Villa' : 10252,
    'Brentford' : 9937, 
    'Bournemouth' : 8678, 
    'Brighton': 10204,
    'Chelsea' : 8455,
    'Crystal Palace' : 9826, 
    'Everton' : 8668, 
    'Fulham' : 9879, 
    'Leeds' : 8463, 
    'Leicester' : 8197,
    'Liverpool' : 8650,
    'Nottingham Forest' : 10203,
    'Newcastle' : 10261,
    'Man Utd' : 10260,
    'Man City' : 8456,
    'Southampton' : 8466,
    'Tottenham' : 8586, 
    'West Ham' : 8654,
    'Wolves' : 8602,
    "Burnley": 8191, 
    "Sheff Utd": 8657, 
    "Luton": 8346
}

ids = {
    'Arsenal': 13,
    'Aston Villa' : 24,
    'Brentford' : 189, 
    'Bournemouth' : 183,
    'Brighton': 211,
    'Chelsea' : 15,
    'Crystal Palace' : 162,
    'Everton' : 31, 
    'Fulham' : 170, 
    'Leeds' : 19, 
    'Leicester' : 14,
    'Liverpool' : 26,
    'Nottingham Forest' : 174,
    'Newcastle' : 23,
    'Man Utd' : 32,
    'Man City' : 167,
    'Southampton' : 18,
    'Tottenham' : 30, 
    'West Ham' : 29,
    'Wolves' : 161,
    'Burnley' : 184, 
    'Sheff Utd' : 163, 
    'Luton' : 95
}

def colorFader(c1,c2,mix=0): 
    c1=np.array(mcolors.to_rgb(c1))
    c2=np.array(mcolors.to_rgb(c2))
    
    mix = max(0.35, mix)
    
    return mcolors.to_hex((mix)*c1 + (1-mix)*c2)


def df_transform(df, team):

    for col in ['field_tilt', 'opp_buildup']:
        df[col] *= 100

    df_ranks = df.copy()
    for col in df_ranks.iloc[:, 1:]:
        if col in ['npxg_allowed_op', 'npxg_allowed_sp', 'xt_allowed', 'box_shots_conceded', 'ppda', 
                   'opp_buildup']:
            df_ranks[col] = df_ranks[col].rank(ascending=False)
        else:
            df_ranks[col] = df_ranks[col].rank(ascending=True)

        df_ranks[col] = df_ranks[col]/len(df_ranks)

    df = df[df['Team'] == team].reset_index(drop=True).iloc[:, 1:]
    df_ranks = df_ranks[df_ranks['Team'] == team].reset_index(drop=True).iloc[:, 1:]
    
    order = ['npxg_created_op', 'npxg_created_sp', 'xt_created', 'box_shots', 'box_touches',
             'npxg_allowed_op','npxg_allowed_sp',  'xt_allowed', 'box_shots_conceded', 
             'comp_passes', 'avg_seq_len', 'field_tilt',
             'ppda', 'opp_buildup','high_pressure_regains', 'def_action_height']
    
    df = df[order]
    df_ranks = df_ranks[order]
    
    df.columns = ['NPXG Created -\nOpen Play\n', 'NPXG Created -\nSet Piece\n', 
                  'XT Created', 
                  'Box Shots\n', 'Box Touches\n', 
                  '\nNPXG Conceded -\nOpen Play', '\nNPXG Conceded -\nSet Piece', 
                  'XT Conceded', 
                  '\nBox Shots\nConceded', 
                  '\nCompleted\nPasses', '\nAverage Seq\nLength', '\nField Tilt',
                  'PPDA\n', 'Opp Buildup %\n', 'High Regains\n', 'Def Action\nHeight\n']

    
    return df, df_ranks


def plotter(df_team1, df_team1_ranks, team1, 
            df_team2, df_team2_ranks, team2,
            color1, color2):
    
    fields = df_team1.columns.tolist()
    
    values = df_team1_ranks.loc[0, :].values.flatten().tolist()
    values = [round(val,2) for val in values]
    values2 = df_team1.loc[0, :].values.flatten().tolist()
    values2 = [round(val,2) for val in values2]
    
    
    values3 = df_team2_ranks.loc[0, :].values.flatten().tolist()
    values3 = [round(val,2) for val in values3]
    values4 = df_team2.loc[0, :].values.flatten().tolist()
    values4 = [round(val,2) for val in values4]
    
    fig, axs = grid(figheight=14, grid_height=0.875, title_height=0.1, endnote_height=0.025,
                title_space=0, endnote_space=0, grid_key='radar', axis=False)
    
    
    low = [0] * len(fields)
    hi = [1] * len(fields)
    
    radar = Radar(fields, low, hi,
                  num_rings=4, 
                  ring_width=1, 
                  center_circle_radius=1)


    radar.setup_axis(ax=axs['radar'], facecolor='#2B2B2B')

    rings_inner = radar.draw_circles(ax=axs['radar'], facecolor='#2B2B2B', edgecolor='white', alpha=0.4, lw=1.5)

    radar_output = radar.draw_radar_compare(values, values3,  ax=axs['radar'],
                                            kwargs_radar={'facecolor': color1, 'alpha':0.55},
                                            kwargs_compare={'facecolor': color2, 'alpha': 0.6})

    radar_poly, radar_poly2, vertices1, vertices2 = radar_output

    col_labels = radar.draw_param_labels(ax=axs['radar'],color="white", fontsize=18, fontname = 'Sans Serif')

    rot = 360
    for i in range(len(vertices1)):
        rot = round(360-((360/len(fields))*i),0)
        if rot in range(90, 270):
            rot = rot - 180 

        x,y = vertices1[i]
        val = values2[i]
        axs['radar'].annotate(xy = (x,y), text = val, rotation=rot,
                              bbox=dict(facecolor= color1, edgecolor='white', boxstyle='round', alpha=1), 
                              color='white', fontname = 'Sans Serif', fontsize = 15)


    rot = 360
    for i in range(len(vertices2)):
        rot = round(360-((360/len(fields))*i),0)
        if rot in range(90, 270):
            rot = rot - 180 

        x,y = vertices2[i]
        val = values4[i]
        axs['radar'].annotate(xy = (x,y), text = val, rotation=rot,
                              bbox=dict(facecolor= color2, edgecolor='white', boxstyle='round', alpha=1), 
                              color='white', fontname = 'Sans Serif', fontsize = 15)

    title1_text = axs['title'].text(0.15, 0.62, team1.upper(), fontsize=25, fontname = 'Sans Serif',
                                ha='left', va='center', color='white')

    title3_text = axs['title'].text(0.87, 0.62, team2.upper(), fontsize=25, fontname = 'Sans Serif',
                                    ha='right', va='center', color='white')

    axs['title'].axhline(y = 0.4, xmin = 0.05, xmax = 0.5, color=color1, lw=3) 
    axs['title'].axhline(y = 0.4, xmin = 0.5, xmax = 0.95, color=color2, lw=3) 

    endnote_text = axs['endnote'].text(0.8, 0.5, 'CREATED BY @JoeW_32', fontsize=15,
                                       fontname = 'Sans Serif', ha='left', va='center', color='white')
    
    # Badge
    fotmob_url = 'https://images.fotmob.com/image_resources/logo/teamlogo/'
    logo_code = logos[team1]
    club_icon = Image.open(urllib.request.urlopen(f'{fotmob_url}{logo_code:.0f}.png'))
    newax = fig.add_axes([0.04, 0.88, 0.12, 0.12], anchor='NE', zorder=3)
    newax.imshow(club_icon)
    newax.axis('off')
    
    
    # Badge
    fotmob_url = 'https://images.fotmob.com/image_resources/logo/teamlogo/'
    logo_code = logos[team2]
    club_icon = Image.open(urllib.request.urlopen(f'{fotmob_url}{logo_code:.0f}.png'))
    newax2 = fig.add_axes([0.85, 0.88, 0.12, 0.12], anchor='NE', zorder=4)
    newax2.imshow(club_icon)
    newax2.axis('off')

    fig.set_facecolor('#2B2B2B')

    return fig


st.title('Team Comparison')

st.divider()

st.markdown(
"""
- Only includes PL Teams for the 23/24 season as of the minute.
- Data as of the latest Gameweek.
"""
)

df_start1 = pd.read_csv('PL_Teams_2324.csv').iloc[:, 1:]
df_start2 = pd.read_csv('PL_Teams_2324.csv').iloc[:, 1:]

teams  = sorted(df_start1['Team'].tolist())

team1 = st.selectbox(
    'Team Name', 
    teams
)

df_team1, df_team1_ranks = df_transform(df_start1, team1)

teams_compare = [team for team in teams if team != team1]

team2 = st.selectbox(
    'Team Name', 
    teams_compare
)

df_team2, df_team2_ranks = df_transform(df_start2, team2)

color1 = st.color_picker('First Player Colour', '#1A78CF')
color2 = st.color_picker('Second Player Colour', '#D70232')


plot = st.button('Plot Comparison')

if plot:

    fig = plotter(df_team1, df_team1_ranks, team1, 
                  df_team2, df_team2, team2, 
                  color1, color2) 
    
    st.write(fig)

    b = io.BytesIO()
    fig.savefig(b, format='png', bbox_inches="tight")

    btn = st.download_button(
        label="Download Plot",
        data=b,
        file_name= team1.replace(' ', '') + "vs" + team2.replace(' ', '') + "Plot.png",
        mime="image/png"
    )

