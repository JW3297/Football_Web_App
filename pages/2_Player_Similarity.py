# General
import pandas as pd
import numpy as np
import io

# Plotting
from mplsoccer import Radar, grid

# ML
from sklearn import preprocessing
from sklearn.decomposition import PCA

# Dashboard
import streamlit as st


def data_prep_posMins(df, position):
    
    df_pos = df[(df['Position'] == position)].reset_index(drop=True)

    col_90s = df_pos['Mins']/90
    for col in df_pos.iloc[:, 4:]:
        if col == 'Pass Completion Rate':
            continue 

        df_pos[col] = df_pos[col]/col_90s

    df_ranks = df_pos.copy()

    for col in df_ranks.iloc[:, 4:]:
        if col == 'Fouls':
            df_ranks[col] = df_ranks[col].rank(ascending=False)
        else:
            df_ranks[col] = df_ranks[col].rank(ascending=True)

        df_ranks[col] = df_ranks[col]/len(df_ranks)
        
    return df_pos, df_ranks


def data_prep_allMins(df, player, position):
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
    for col in df_pos.iloc[:, 4:]:
        if col == 'Pass Completion Rate':
            continue 

        df_pos[col] = df_pos[col]/col_90s

    df_ranks = df_pos.copy()

    for col in df_ranks.iloc[:, 4:]:
        if col == 'Fouls':
            df_ranks[col] = df_ranks[col].rank(ascending=False)
        else:
            df_ranks[col] = df_ranks[col].rank(ascending=True)

        df_ranks[col] = df_ranks[col]/len(df_ranks)

        
    return df_pos, df_ranks

def playerSimilaritySearch(df, name):
    df_info, df_metrics = df.iloc[:, :4], df.iloc[:, 4:]
    cols = df_metrics.columns.tolist()

    vals = df_metrics.values
    scaler = preprocessing.MinMaxScaler()
    vals_scaled = scaler.fit_transform(vals)
    vals_norm = pd.DataFrame(vals_scaled, columns=cols)
    
    pca_2d = PCA(n_components = 2)
    df_reduced = pd.DataFrame(pca_2d.fit_transform(vals_norm), columns=['x', 'y'])
    df = pd.concat([df_info, df_reduced], axis=1)

    x,y = df[df['playerName'] == name]['x'].tolist()[0], df[df['playerName'] == name]['y'].tolist()[0]

    df['Difference Score'] = round(np.sqrt((x - df['x'])**2 + (y - df['y'])**2), 3)
    df = df.sort_values(by='Difference Score').reset_index(drop=True)[['playerName', 'Mins', 'Difference Score']]
    df.columns = ['Player', 'Minutes', 'Difference Score']

    return df.head(10)

def df_player(df_pos, df_ranks, name):
    df_player_rank = df_ranks[df_ranks['playerName'] == name].reset_index(drop=True)
    df_player_vals = df_pos[df_pos['playerName'] == name].reset_index(drop=True)
    
    df_player_vals['Pass Completion Rate'] *= 100
    mins = df_player_rank['Mins'][0]
    
    return df_player_rank, df_player_vals, mins

def plotter(df_player_rank1, df_player_vals1, name1, pos, mins1,
            df_player_rank2, df_player_vals2, name2, mins2):
    
    if pos == 'Center Mid':
        cols = [
                'Passes Attempted', 'Pass Completion Rate', 'Forward Passes','Long Passes', 'Progressive Passes',
                'npxG', 'xA', 'Shots',
                'Tackles', 'Interceptions', 'Fouls','Aerials',
                'Touches', 'xT Received','Progressive Carries'
               ]
        df_player_rank1 = df_player_rank1[cols]
        df_player_vals1 = df_player_vals1[cols]
        df_player_rank2 = df_player_rank2[cols]
        df_player_vals2 = df_player_vals2[cols]
        
        fields = [
                'Passes \nAttempted', 'Pass \nCompletion Rate', 'Forward \nPasses', 'Long \nPasses', 'Progressive \nPasses',
                'npxG', 'xA', 'Shots',
                'Tackles', 'Interceptions', 'Fouls','Aerials',
                'Touches', 'xT Received','Progressive \nCarries'
               ]   
        
    
    elif pos == 'Attacking Mid/Winger':
        cols = [
                'Pass Completion Rate', 'Forward Passes','Progressive Passes','Box Passes',
                'npxG', 'xA', 'Shots',
                'Tackles', 'Fouls', 'Recoveries',
                'Box Touches', 'Final 1/3 Touches', 'xT Received','Progressive Carries', 'Successful Take Ons'
               ]
        df_player_rank1 = df_player_rank1[cols]
        df_player_vals1 = df_player_vals1[cols]
        df_player_rank2 = df_player_rank2[cols]
        df_player_vals2 = df_player_vals2[cols]
        
        
        fields = [
                'Pass \nCompletion Rate', 'Forward \nPasses','Progressive \nPasses','Box \nPasses',
                'npxG', 'xA', 'Shots',
                'Tackles', 'Fouls', 'Recoveries',
                'Box \nTouches', 'Final 1/3 \nTouches', 'xT Received','Progressive \nCarries', 'Successful \nTake Ons'
               ]   
        
        
        
    elif pos == 'Full Back':
        cols = [
                'Pass Completion Rate', 'Forward Passes','Progressive Passes',
                'npxG', 'xA',
                'Tackles', 'Interceptions', 'Fouls', 'Aerials', 'Recoveries',
                'Touches', 'Final 1/3 Touches', 'Progressive Carries', 'Successful Take Ons'
               ]
        
        df_player_rank1 = df_player_rank1[cols]
        df_player_vals1 = df_player_vals1[cols]
        df_player_rank2 = df_player_rank2[cols]
        df_player_vals2 = df_player_vals2[cols]
        
        
        fields = [
                'Pass \nCompletion Rate', 'Forward \nPasses','Progressive \nPasses',
                'npxG', 'xA',
                'Tackles', 'Interceptions', 'Fouls', 'Aerials', 'Recoveries',
                'Touches', 'Final 1/3 \nTouches', 'Progressive \nCarries', 'Successful \nTake Ons'
               ]
        
        
    elif pos == 'Forward':
        cols = [
                'Pass Completion Rate', 'Progressive Passes', 'Box Passes',
                'npxG', 'xA', 'Shots', 
                'Fouls', 'Aerials', 'Recoveries',
                'Box Touches', 'Final 1/3 Touches', 'xT Received','Progressive Carries', 'Successful Take Ons'
               ]
        
        df_player_rank1 = df_player_rank1[cols]
        df_player_vals1 = df_player_vals1[cols]
        df_player_rank2 = df_player_rank2[cols]
        df_player_vals2 = df_player_vals2[cols]
        
        fields = [
                'Pass \nCompletion Rate', 'Progressive \nPasses', 'Box \nPasses',
                'npxG', 'xA', 'Shots',
                'Fouls', 'Aerials', 'Recoveries',
                'Box \nTouches', 'Final 1/3 \nTouches', 'xT Received','Progressive \nCarries', 'Successful \nTake Ons'
               ]


    elif pos == 'Center Back':
        cols = [
                'Passes Attempted', 'Pass Completion Rate', 'Long Passes', 'Progressive Passes',
                'npxG', 'xA',
                'Tackles', 'Interceptions', 'Fouls', 'Aerials', 'Recoveries',
                'Touches', 'Progressive Carries'
               ]
        
        df_player_rank1 = df_player_rank1[cols]
        df_player_vals1 = df_player_vals1[cols]
        df_player_rank2 = df_player_rank2[cols]
        df_player_vals2 = df_player_vals2[cols]
        
        
        fields = [
                'Passes \nAttempted', 'Pass \nCompletion Rate', 'Long \nPasses', 'Progressive \nPasses',
                'npxG', 'xA',
                'Tackles', 'Interceptions', 'Fouls', 'Aerials', 'Recoveries',
                'Touches', 'Progressive \nCarries',
               ]
        
    
    fields = [field.upper() for field in fields]
    
    values = df_player_rank1.loc[0, :].values.flatten().tolist()
    values = [round(val,2) for val in values]
    values2 = df_player_vals1.loc[0, :].values.flatten().tolist()
    values2 = [round(val,2) for val in values2]
    
    
    values3 = df_player_rank2.loc[0, :].values.flatten().tolist()
    values3 = [round(val,2) for val in values3]
    values4 = df_player_vals2.loc[0, :].values.flatten().tolist()
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
                                            kwargs_radar={'facecolor': '#1A78CF', 'alpha':0.55},
                                            kwargs_compare={'facecolor': '#D70232', 'alpha': 0.6})

    radar_poly, radar_poly2, vertices1, vertices2 = radar_output

    col_labels = radar.draw_param_labels(ax=axs['radar'],color="white", fontsize=18, fontname = 'Sans Serif')

    rot = 360
    for i in range(len(vertices1)):
        rot = 360-((360/len(fields))*i)
        if rot in range(90, 270):
            rot = rot - 180 

        x,y = vertices1[i]
        val = values2[i]
        axs['radar'].annotate(xy = (x,y), text = val, rotation=rot,
                              bbox=dict(facecolor= '#1A78CF', edgecolor='white', boxstyle='round', alpha=1), 
                              color='white', fontname = 'Sans Serif', fontsize = 15)


    rot = 360
    for i in range(len(vertices2)):
        rot = 360-((360/len(fields))*i)
        if rot in range(90, 270):
            rot = rot - 180 

        x,y = vertices2[i]
        val = values4[i]
        axs['radar'].annotate(xy = (x,y), text = val, rotation=rot,
                              bbox=dict(facecolor= '#D70232', edgecolor='white', boxstyle='round', alpha=1), 
                              color='white', fontname = 'Sans Serif', fontsize = 15)

    title1_text = axs['title'].text(0.02, 0.85, name1.upper(), fontsize=25, fontname = 'Sans Serif',
                                ha='left', va='center', color='white')
    title1_text = axs['title'].text(0.02, 0.6, str(mins1) + ' MINS', fontsize=17, fontname = 'Sans Serif',
                                ha='left', va='center', color='white')
    title3_text = axs['title'].text(0.98, 0.85, name2.upper(), fontsize=25, fontname = 'Sans Serif',
                                    ha='right', va='center', color='white')
    title4_text = axs['title'].text(0.98, 0.6, str(mins2) + ' MINS', fontsize=17, fontname = 'Sans Serif',
                                    ha='right', va='center', color='white')
    axs['title'].axhline(y = 0.4, xmin = 0.02, xmax = 0.5, color='#1A78CF', lw=3) 
    axs['title'].axhline(y = 0.4, xmin = 0.5, xmax = 0.98, color='#D70232', lw=3) 

    endnote_text = axs['endnote'].text(0.8, 0.5, 'TEMPLATE: ' + position + '\n\nCREATED BY @JoeW_32', fontsize=12,
                                       fontname = 'Sans Serif', ha='left', va='center', color='white')

    fig.set_facecolor('#2B2B2B')

    
    return fig



st.title('Player Similarity Search')

st.divider()

st.markdown(
"""
- Only includes PL Teams for the 23/24 season as of the minute.
- Data as of Gameweek 27.
- PCA is used to identify similar players using a range of calculated metrics. 
"""
)


df = pd.read_csv('player_db.csv').iloc[:,1:]
df = df[(df['Mins'] >= 100) & 
        (df['Position'] != 'Goalkeeper')].reset_index(drop=True)

players = sorted(list(set(df['playerName'])))

player1 = st.selectbox(
    'Player Name', 
    players
)

positions = list(df[df['playerName'] == player1]['Position'])

position = st.selectbox(
    'Position', 
    positions
)

all_mins = st.checkbox('To include all minutes at above positions')

if all_mins:
    df_pos1, df_ranks1 = data_prep_allMins(df, player1, position)

else:
    df_pos1, df_ranks1 = data_prep_posMins(df, position)

df_rank1, df_vals1, mins1 = df_player(df_pos1, df_ranks1, player1)


df_similar = playerSimilaritySearch(df_pos1, player1)

generate = st.button('Search')

if "generate_state" not in st.session_state:
    st.session_state.generate_state = False

if generate or st.session_state.generate_state:

    st.session_state.generate_state = True

    st.dataframe(df_similar)

    player_compare = df_similar[df_similar['Player'] != player1]['Player'].unique().tolist()

    player2 = st.selectbox(
        'Player To Compare', 
        player_compare
    )

    df_pos2, df_ranks2 = data_prep_posMins(df, position)
    df_rank2, df_vals2, mins2 = df_player(df_pos2, df_ranks2, player2) 

    plot = st.button('Plot Comparison')

    if plot:

        fig = plotter(df_rank1, df_vals1, player1, position, mins1,
                        df_rank2, df_vals2, player2, mins2) 
        
        st.write(fig)

        b = io.BytesIO()
        fig.savefig(b, format='png', bbox_inches="tight")

        btn = st.download_button(
            label="Download Plot",
            data=b,
            file_name= player1.replace(' ', '') + "vs" + player2.replace(' ', '') + "Plot.png",
            mime="image/png"
                )

