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


def df_transform(df, team, season):

    df_ranks = df.copy()
    for col in df_ranks.iloc[:, 2:]:
        if col in ['npxg_allowed_op', 'npxg_allowed_sp', 'xt_allowed', 'box_shots_conceded', 'ppda', 
                   'opp_buildup']:
            df_ranks[col] = df_ranks[col].rank(ascending=False)
        else:
            df_ranks[col] = df_ranks[col].rank(ascending=True)

        df_ranks[col] = df_ranks[col]/len(df_ranks)

    df = df[(df['Team'] == team) & (df['Season'] == season)].reset_index(drop=True).iloc[:, 1:]
    df_ranks = df_ranks[(df_ranks['Team'] == team) & (df_ranks['Season'] == season)].reset_index(drop=True).iloc[:, 1:]
    
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


def plotter(df_team1, df_team1_ranks, team1, team1_season,
            df_team2, df_team2_ranks, team2, team2_season,
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
        
    if team1 == 'Nottingham Forest':
        name1 = 'Forest'
    else:
        name1 = team1

    if team2 == 'Nottingham Forest':
        name2 = 'Forest'
    else:
        name2 = team2

    title1_text = axs['title'].text(0.15, 0.62, name1.upper() + ' - ' + team1_season[2:], fontsize=25, fontname = 'Sans Serif',
                                ha='left', va='center', color='white')

    title3_text = axs['title'].text(0.86, 0.62, name2.upper() + ' - ' + team2_season[2:], fontsize=25, fontname = 'Sans Serif',
                                    ha='right', va='center', color='white')

    axs['title'].axhline(y = 0.4, xmin = 0.06, xmax = 0.5, color=color1, lw=3) 
    axs['title'].axhline(y = 0.4, xmin = 0.5, xmax = 0.94, color=color2, lw=3) 

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
- Includes PL Teams for the 22/23 and 23/24 season as of the minute.
- Will add 24/25 data as the season progresses.
"""
)

df = pd.read_csv('PL_Teams.csv').iloc[:, 1:]

seasons = sorted(list(set(df['Season'])))

team1_season = st.selectbox(
    'Team 1 Season', 
    seasons
)

df_start1 = df
df_start2 = df

teams = sorted(df[df['Season'] == team1_season]['Team'].tolist())

team1 = st.selectbox(
    'Team Name', 
    teams
)

df_team1, df_team1_ranks = df_transform(df_start1, team1, team1_season)

team2_season = st.selectbox(
    'Team 2 Season', 
    seasons
)

if team1_season == team2_season:
    teams_compare = [team for team in teams if team != team1]
else:
    teams_compare = sorted(df[df['Season'] == team2_season]['Team'].tolist())

team2 = st.selectbox(
    'Team Name', 
    teams_compare
)

df_team2, df_team2_ranks = df_transform(df_start2, team2, team2_season)

color1 = st.color_picker('First Team Colour', '#1A78CF')
color2 = st.color_picker('Second Team Colour', '#D70232')


plot = st.button('Plot Comparison')

if plot:

    fig = plotter(df_team1, df_team1_ranks, team1, team1_season,
                  df_team2, df_team2_ranks, team2, team2_season,
                  color1, color2) 
    
    st.write(fig)

    b = io.BytesIO()
    fig.savefig(b, format='png', bbox_inches="tight")

    btn = st.download_button(
        label="Download Plot",
        data=b,
        file_name= team1.replace(' ', '') + team1_season[2:] + "vs" + team2.replace(' ', '') + team2_season[2:] + "Plot.png",
        mime="image/png"
    )

