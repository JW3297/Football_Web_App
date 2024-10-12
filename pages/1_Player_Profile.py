# General
import pandas as pd
import numpy as np
import io

# Plotters
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mplsoccer import PyPizza
import matplotlib as mpl

# Dashboard
import streamlit as st


def df_breakdown(df, position, mins=0):
    
    df_pos = df[(df['Position'] == position) & 
                (df['Mins'] >= mins)].reset_index(drop=True)

    col_90s = df_pos['Mins']/90
    for col in df_pos.iloc[:, 5:]:
        if col == 'Pass Completion Rate':
            continue 

        df_pos[col] = df_pos[col]/col_90s

    df_ranks = df_pos.copy()

    for col in df_ranks.iloc[:, 5:]:
        if col == 'Fouls':
            df_ranks[col] = df_ranks[col].rank(ascending=False)
        else:
            df_ranks[col] = df_ranks[col].rank(ascending=True)

        df_ranks[col] = df_ranks[col]/len(df_ranks)
        
    return df_pos, df_ranks


def df_inc_all(df, player, position):
    df1 = df[(df['Position'] == position) & 
             (df['playerName'] != player)].reset_index(drop=True)

    df2 = df[df['playerName'] == player].reset_index(drop=True)
    
    cols = ['Mins', 'Passes Completed', 'Passes Attempted', 'Forward Passes', 'Long Passes', 
            'xT via Pass', 'npxG', 'xA', 'Shots', 'Tackles', 'Interceptions', 'Fouls', 'Aerials', 'Recoveries', 
            'Ball Wins', 'Touches', 'Box Touches', 'Final 1/3 Touches', 'Progressive Passes', 
            'Progressive Carries', 'Box Passes', 'Box Carries', 'Successful Take Ons', 'xT Carry', 'xT Received']
    
    d1 = dict.fromkeys(cols, 'sum')
    df2 = df2.groupby(['playerName', 'PlayerID']) \
             .agg(d1) \
             .reset_index()

    df2['Pass Completion Rate'] = df2['Passes Completed'] / df2['Passes Attempted']
    df2['Position'] = position
    order = list(df2.columns[:2]) + [df2.columns[28]] + list(df2.columns[2:6]) + [df2.columns[27]] + list(df2.columns[6:27])

    df2 = df2[order]
    
    df_pos = pd.concat([df1, df2], axis=0).reset_index(drop=True)
    
    col_90s = df_pos['Mins']/90
    for col in df_pos.iloc[:, 5:]:
        if col == 'Pass Completion Rate':
            continue 

        df_pos[col] = df_pos[col]/col_90s

    df_ranks = df_pos.copy()

    for col in df_ranks.iloc[:, 5:]:
        if col == 'Fouls':
            df_ranks[col] = df_ranks[col].rank(ascending=False)
        else:
            df_ranks[col] = df_ranks[col].rank(ascending=True)

        df_ranks[col] = df_ranks[col]/len(df_ranks)
        
    return df_pos, df_ranks
    


def df_player(df_pos, df_ranks, name):
    df_player_rank = df_ranks[df_ranks['playerName'] == name].reset_index(drop=True)
    df_player_vals = df_pos[df_pos['playerName'] == name].reset_index(drop=True)
    
    df_player_vals['Pass Completion Rate'] *= 100
    mins = df_player_rank['Mins'][0]
    
    return df_player_rank, df_player_vals, mins


def colorFader(c1,c2,mix=0): 
    c1=np.array(mcolors.to_rgb(c1))
    c2=np.array(mcolors.to_rgb(c2))
    
    mix = max(0.35, mix)
    
    return mcolors.to_hex((mix)*c1 + (1-mix)*c2)

@st.cache_resource
def plotter(df_player_rank, df_player_vals, name, pos, mins, season):
    
    if pos == 'Center Mid':
        cols = [
                'Passes Attempted', 'Pass Completion Rate', 'Forward Passes','Long Passes', 'Progressive Passes',
                'npxG', 'xA', 'Shots',
                'Tackles', 'Interceptions', 'Fouls','Aerials',
                'Touches', 'xT Received','Progressive Carries'
               ]
        df_player_rank = df_player_rank[cols]
        df_player_vals = df_player_vals[cols]
        
        fields = [
                'Passes \nAttempted', 'Pass \nCompletion Rate', 'Forward \nPasses', 'Long \nPasses', 'Progressive \nPasses',
                'npxG', 'xA', 'Shots',
                'Tackles', 'Interceptions', 'Fouls','Aerials',
                'Touches', 'xT Received','Progressive \nCarries'
               ]   
        
        blank_colors = ["#848484"] * 15
        slice_colors = ["#1A78CF"] * 5 + ["#D70232"] * 3 + ["#228B22"] * 4 +  ['#FF8000'] * 3
        text_colors = ["#000000"] * 15
        
    
    elif pos == 'Attacking Mid/Winger':
        cols = [
                'Pass Completion Rate', 'Forward Passes','Progressive Passes','Box Passes',
                'npxG', 'xA', 'Shots',
                'Tackles', 'Fouls', 'Recoveries',
                'Box Touches', 'Final 1/3 Touches', 'xT Received','Progressive Carries', 'Successful Take Ons'
               ]
        df_player_rank = df_player_rank[cols]
        df_player_vals = df_player_vals[cols]
        
        fields = [
                'Pass \nCompletion Rate', 'Forward \nPasses','Progressive \nPasses','Box \nPasses',
                'npxG', 'xA', 'Shots',
                'Tackles', 'Fouls', 'Recoveries',
                'Box \nTouches', 'Final 1/3 \nTouches', 'xT Received','Progressive \nCarries', 'Successful \nTake Ons'
               ]   
        
        blank_colors = ["#848484"] * 15
        slice_colors = ["#1A78CF"] * 4 + ["#D70232"] * 3 + ["#228B22"] * 3 +  ['#FF8000'] * 5
        text_colors = ["#000000"] * 15
        
        
    elif pos == 'Full Back':
        cols = [
                'Pass Completion Rate', 'Forward Passes','Progressive Passes',
                'npxG', 'xA',
                'Tackles', 'Interceptions', 'Fouls', 'Aerials', 'Recoveries',
                'Touches', 'Final 1/3 Touches', 'Progressive Carries', 'Successful Take Ons'
               ]
        df_player_rank = df_player_rank[cols]
        df_player_vals = df_player_vals[cols]
        
        fields = [
                'Pass \nCompletion Rate', 'Forward \nPasses','Progressive \nPasses',
                'npxG', 'xA',
                'Tackles', 'Interceptions', 'Fouls', 'Aerials', 'Recoveries',
                'Touches', 'Final 1/3 \nTouches', 'Progressive \nCarries', 'Successful \nTake Ons'
               ]
        
        blank_colors = ["#848484"] * 14
        slice_colors = ["#1A78CF"] * 3 + ["#D70232"] * 2 + ["#228B22"] * 5 +  ['#FF8000'] * 4
        text_colors = ["#000000"] * 14
        
    elif pos == 'Forward':
        cols = [
                'Pass Completion Rate', 'Progressive Passes', 'Box Passes',
                'npxG', 'xA', 'Shots', 
                'Fouls', 'Aerials', 'Recoveries',
                'Box Touches', 'Final 1/3 Touches', 'xT Received','Progressive Carries', 'Successful Take Ons'
               ]
        df_player_rank = df_player_rank[cols]
        df_player_vals = df_player_vals[cols]
        
        fields = [
                'Pass \nCompletion Rate', 'Progressive \nPasses', 'Box \nPasses',
                'npxG', 'xA', 'Shots',
                'Fouls', 'Aerials', 'Recoveries',
                'Box \nTouches', 'Final 1/3 \nTouches', 'xT Received','Progressive \nCarries', 'Successful \nTake Ons'
               ]
        
        blank_colors = ["#848484"] * 14
        slice_colors = ["#1A78CF"] * 3 + ["#D70232"] * 3 + ["#228B22"] * 3 +  ['#FF8000'] * 5
        text_colors = ["#000000"] * 14

    elif pos == 'Center Back':
        cols = [
                'Passes Attempted', 'Pass Completion Rate', 'Long Passes', 'Progressive Passes',
                'npxG', 'xA',
                'Tackles', 'Interceptions', 'Fouls', 'Aerials', 'Recoveries',
                'Touches', 'Progressive Carries'
               ]
        df_player_rank = df_player_rank[cols]
        df_player_vals = df_player_vals[cols]
        
        fields = [
                'Passes \nAttempted', 'Pass \nCompletion Rate', 'Long \nPasses', 'Progressive \nPasses',
                'npxG', 'xA',
                'Tackles', 'Interceptions', 'Fouls', 'Aerials', 'Recoveries',
                'Touches', 'Progressive \nCarries',
               ]
        
        blank_colors = ["#848484"] * 13
        slice_colors = ["#1A78CF"] * 4 + ["#D70232"] * 2 + ["#228B22"] * 5 +  ['#FF8000'] * 2
        text_colors = ["#000000"] * 13
        
    fields = [field.upper() for field in fields]

    values = df_player_rank.loc[0, :].values.flatten().tolist()
    values = [round(val*100,2) for val in values]

    values2 = df_player_vals.loc[0, :].values.flatten().tolist()
    values2 = [round(val,2) for val in values2]
    
    slice_colors2 = []
    alt_color = '#2B2B2B'
    for i, color in enumerate(slice_colors):
        pct = values[i]

        slice_colors2.append(colorFader(color, '#2B2B2B', pct/100))
        
        
    mpl.rcParams['figure.dpi'] = 600

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
        figsize=(7, 9),                     # adjust the figsize according to your need
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
        0.1, 0.97, name.replace("-"," ").upper() , size=15,
        ha="left",color="white",fontname = 'Sans Serif'
    )

    # add subtitle
    fig.text(
        0.1, 0.9425,
        str(mins) + ' MINUTES | ' + pos.upper(),
        size=9,
        ha="left",color="white"
    )


    # add subtitle
    fig.text(
        0.1, 0.9175,
        "Percentile Rank vs Other PL Players | " + season + " | Created by ".upper() + '@JoeW__32',
        size=9,
        ha="left",color="white"
    )

    # add text
    fig.text(
        0.13, 0.89, "Passing         Attacking        Defending       Receptions/Carrying".upper(), size=9,
        color="white"
    )

    fig.patches.extend([
        plt.Rectangle(
            (0.1, 0.885), 0.023, 0.019, fill=True, color="#1A78CF",
            transform=fig.transFigure, figure=fig
        ),
        plt.Rectangle(
            (0.23, 0.885), 0.023, 0.019, fill=True, color="#D70232",
            transform=fig.transFigure, figure=fig
        ),
        plt.Rectangle(
            (0.37, 0.885), 0.023, 0.019, fill=True, color="#228B22",
            transform=fig.transFigure, figure=fig
        ),
        plt.Rectangle(
            (0.52, 0.885), 0.023, 0.019, fill=True, color="#FF8000",
            transform=fig.transFigure, figure=fig
        ),

    ])



    for i, text in enumerate(plot.get_value_texts()):
        text.set_text(values2[i])


    return fig


st.title('Player Profiles')

st.divider()

st.markdown(
"""
- Includes PL Teams for the 22/23 and 23/24 season as of the minute.
- 24/25 data as of Gameweek 7.
- Only includes players who have played more than 100 minutes in the respective position. 
- Goalkeepers not included. 
"""
)


df = pd.read_csv('player_db_combined.csv').iloc[:,1:]

df = df[(df['Mins'] >= 90) & 
        (df['Position'] != 'Goalkeeper')].reset_index(drop=True)

seasons = sorted(list(set(df['Season'])))

season = st.selectbox(
    'Season', 
    seasons
)

df = df[df['Season'] == season].reset_index(drop=True)

players = sorted(list(set(df['playerName'])))

player = st.selectbox(
    'Player Name', 
    players
)

positions = list(df[df['playerName'] == player]['Position'])

position = st.selectbox(
    'Position', 
    positions
)

all_mins = st.checkbox('To include all minutes at above positions')

if all_mins:
    df_pos, df_ranks = df_inc_all(df, player, position)

else:
    df_pos, df_ranks = df_breakdown(df, position)

df_player_rank, df_player_vals, mins =  df_player(df_pos, df_ranks, player)


generate = st.button('Create Plot')

if generate:
    fig = plotter(df_player_rank, df_player_vals, player, position, mins, season)
    st.write(fig)

    b = io.BytesIO()
    fig.savefig(b, format='png', bbox_inches="tight")

    btn = st.download_button(
        label="Download Plot",
        data=b,
        file_name= player.replace(' ', '') +  "Plot.png",
        mime="image/png"
        )


