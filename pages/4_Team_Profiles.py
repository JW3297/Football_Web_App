# General
import pandas as pd
import numpy as np
import io

# Plotting
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import matplotlib.colors as mcolors
from mplsoccer import PyPizza, add_image, FontManager
import matplotlib as mpl
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
    "Luton": "#FB6B07", 
    "Ipswich" : '#2D5DE7'
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
    "Luton": 8346,
    "Ipswich" : 9902
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
    'Luton' : 95, 
    "Ipswich" : 165
}

# Color Map
def colorFader(c1,c2,mix=0): 
    c1=np.array(mcolors.to_rgb(c1))
    c2=np.array(mcolors.to_rgb(c2))
    
    mix = max(0.35, mix)
    
    return mcolors.to_hex((mix)*c1 + (1-mix)*c2)


def df_transform(df, team):

    df_ranks = df.copy()
    for col in df_ranks.iloc[:, 2:]:
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


def plotter(df, df_ranks, name, season):

    fields = df.columns.values.tolist()
    fields = [field.upper() for field in fields]

    values = df_ranks.loc[0, :].values.flatten().tolist()
    values = [round(val*100,2) for val in values]

    values2 = df.loc[0, :].values.flatten().tolist()
    values2 = [round(val,2) for val in values2]

    blank_colors = ["#848484"] * 16
    slice_colors = ["#1A78CF"] * 5 + ["#D70232"] * 4 + ["#228B22"] * 3 +  ['#FF8000'] * 4 
    text_colors = ["#000000"] * 16

    slice_colors2 = []
    alt_color = '#2B2B2B'
    for i, color in enumerate(slice_colors):
        pct = values[i]

        slice_colors2.append(colorFader(color, '#2B2B2B', pct/100))

    mpl.rcParams['figure.dpi'] = 400

    plot = PyPizza(
        inner_circle_size = 20,
        params=fields,                  # list of parameters
        straight_line_color="#2B2B2B",  # color for straight lines
        straight_line_lw=1,             # linewidth for straight lines
        other_circle_lw=1,              # linewidth for other circles
        other_circle_ls= '--'  ,        # linestyle for other circles
        last_circle_lw=1,               # linewidth of last circle
        last_circle_ls = '-',
        background_color = '#2B2B2B',
        straight_line_limit = 101
    )

    fig, ax = plot.make_pizza(
        values,                             # list of values
        figsize=(8, 9),                     # adjust the figsize according to your need
        slice_colors=slice_colors2,          # color for individual slices
        value_colors=text_colors,           # color for the value-text
        value_bck_colors=slice_colors2,      # color for the blank spaces
        blank_alpha=1 ,                     # alpha for blank-space colors

        kwargs_slices=dict(
            edgecolor="#2B2B2B", zorder=3, linewidth=2
        ),                                  # values to be used when plotting slices

        kwargs_params=dict(
            color="white", fontsize=9, fontname = 'Sans Serif',
            va="center"
        ),                                  # values to be used when adding parameter labels

        kwargs_values=dict(
            color="white", fontsize=9, fontname = 'Sans Serif',
            zorder=5,
            bbox=dict(
                edgecolor="#2B2B2B", facecolor="white",
                boxstyle="round,pad=.2", lw=1
            )
        )                                    # values to be used when adding parameter-values labels    
    )


    fig.text(
        0.25, 0.975, name.replace("-"," ").upper() , size=15,
        ha="left",color="white",fontname = 'Sans Serif'
    )

    # add subtitle
    fig.text(
        0.25, 0.9475,
        "Percentile Rank vs Other PL Teams | " + str(season) + " | Created by ".upper() + '@JoeW__32',
        size=9,
        ha="left",color="white"
    )


    # add text
    fig.text(
        0.28, 0.92, "Attacking        Defending        Possession        Pressing".upper(), size=9,
        color="white"
    )

    fig.patches.extend([
        plt.Rectangle(
            (0.25, 0.915), 0.023, 0.019, fill=True, color="#1A78CF",
            transform=fig.transFigure, figure=fig
        ),
        plt.Rectangle(
            (0.38, 0.915), 0.023, 0.019, fill=True, color="#D70232",
            transform=fig.transFigure, figure=fig
        ),
        plt.Rectangle(
            (0.51, 0.915), 0.023, 0.019, fill=True, color="#228B22",
            transform=fig.transFigure, figure=fig
        ),
        plt.Rectangle(
            (0.65, 0.915), 0.023, 0.019, fill=True, color="#FF8000",
            transform=fig.transFigure, figure=fig
        ),

    ])

    for i, text in enumerate(plot.get_value_texts()):
        text.set_text(values2[i])



    # Badge
    fotmob_url = 'https://images.fotmob.com/image_resources/logo/teamlogo/'
    logo_code = logos[name]
    club_icon = Image.open(urllib.request.urlopen(f'{fotmob_url}{logo_code:.0f}.png'))
    newax = fig.add_axes([0.09, 0.875, 0.12, 0.12], anchor='NE', zorder=2)
    newax.imshow(club_icon)
    newax.axis('off')

    # Save
    # fig.savefig('__2324.png', 
    #             bbox_inches="tight",
    #             edgecolor="none",
    #             dpi=500)
    
    return fig

st.title('Team Profiles')

st.divider()

st.markdown(
"""
- Includes PL Teams for the 22/23 and 23/24 season as of the minute.
- 24/25 data as of Gameweek 3.
"""
)

df = pd.read_csv('PL_Teams.csv').iloc[:, 1:]

seasons = sorted(list(set(df['Season'])))

season = st.selectbox(
    'Season', 
    seasons
)

df = df[df['Season'] == season].reset_index(drop=True)

teams  = sorted(df['Team'].tolist())

team = st.selectbox(
    'Team Name', 
    teams
)

df, df_ranks = df_transform(df, team)

generate = st.button('Create Plot')

if generate:
    fig = plotter(df, df_ranks, team, season)
    st.write(fig)

    b = io.BytesIO()
    fig.savefig(b, format='png', bbox_inches="tight")

    btn = st.download_button(
        label="Download Plot",
        data=b,
        file_name= team.replace(' ', '') + season[2:] + "Plot.png",
        mime="image/png"
        )

