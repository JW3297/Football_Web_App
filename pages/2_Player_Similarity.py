import pandas as pd
import numpy as np
import io

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

if generate:
    st.dataframe(df_similar)

    player_compare = df_similar[df_similar['Player'] != player1]['Player'].unique().tolist()

    player2 = st.selectbox(
        'Player To Compare', 
        player_compare
    )

    df_pos2, df_ranks2 = data_prep_posMins(df, position)
    df_rank2, df_vals2, mins2 = df_player(df_pos2, df_ranks2, player2)  

    