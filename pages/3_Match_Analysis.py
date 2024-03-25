# General
import pandas as pd
import numpy as np
from collections import defaultdict
import io


# Football Package
import matchData3 

# Dashboard
import streamlit as st

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
    'Liverpool' : 26,
    'Nottingham Forest' : 174,
    'Newcastle' : 23,
    'Man Utd' : 32,
    'Man City' : 167,
    'Tottenham' : 30, 
    'West Ham' : 29,
    'Wolves' : 161,
    'Burnley' : 184, 
    'Sheff Utd' : 163, 
    'Luton' : 95
}

id_inv = {v: k for k, v in ids.items()}

def matches_generator(name):
    
    teamID = ids[name]
    match_ids = df[df['teamId'] == teamID]['matchId'].unique().tolist()

    matches = defaultdict()

    for i, match_id in enumerate(match_ids):
        teamIds = df[df['matchId'] == match_id]['teamId'].unique().tolist()
        oppId = [ID for ID in teamIds if ID != teamID][0]
        oppName = id_inv[oppId]

        home_away_flag = df[(df['matchId'] == match_id) & 
                            (df['teamId'] == teamID)]['h_a'].unique().tolist()[0]

        if home_away_flag == 'h':
            string = 'Gameweek ' + str(i+1) + ' : ' + name + ' vs ' + oppName
        else:
            string = 'Gameweek ' + str(i+1) + ' : ' + oppName + ' vs ' + name

        matches[string] = match_id
        
    return matches


st.title('Match Analysis')

st.divider()

st.markdown(
"""
- Only includes PL Teams for the 23/24 season as of the minute. 
- Data as of Gameweek 27. 
- Plots can take some time so please be patient. 
"""
)

df = pd.read_csv('pl_2324_matches.csv').iloc[:, 1:]

teams  = sorted([team for team in ids.keys()])

team = st.selectbox(
    'Team', 
    teams
)

matches = matches_generator(team)

match_text = [match for match in matches.keys()]


match = st.selectbox(
    'Match', 
    match_text
)

game_id = matches[match]

events_df = df[df['matchId'] == game_id].reset_index(drop=True)

match_obj = matchData3.FootballMatch(events_df)

plots = ['Match Dashboard', 
         'XG Time Plot',
         'Passing Network', 
         'Defensive Actions', 
         'Zone Control', 
         'Progressive Clusters', 
         'Zone Actions', 
         'XT Zones', 
         'XG Scatter Plot', 
         'XT Scatter Plot', 
         'Field Gains Scatter Plot', 
         'Duels Scatter Plot', 
         'MoM Stats', 
         'MoM Events']

plot = st.selectbox(
    'Plot', 
    plots
)



generate = st.button('Create Plot')

if generate:
    if plot == 'Match Dashboard':
        fig = match_obj.plot_matchDashboard()

    elif plot == 'XG Time Plot':
        fig = match_obj.plot_movingXG()

    elif plot == 'Passing Network':
        fig = match_obj.plot_passNetwork()

    elif plot == 'Defensive Actions':
        fig = match_obj.plot_defensiveActions()

    elif plot == 'Zone Control':
        fig = match_obj.plot_zoneControl()

    elif plot == 'Progressive Clusters':
        fig = match_obj.plot_progClusters()

    elif plot == 'Zone Actions':
        fig = match_obj.plot_zoneActions()

    elif plot == 'XT Zones':
        fig = match_obj.plot_xTZones()

    elif plot == 'XG Scatter Plot':
        fig = match_obj.plot_scatterxG()

    elif plot == 'XT Scatter Plot':
        fig = match_obj.plot_scatterxT()

    elif plot == 'Field Gains Scatter Plot':
        fig = match_obj.plot_scatterFieldGains()

    elif plot == 'Duels Scatter Plot':
        fig = match_obj.plot_duelsWon()

    elif plot == 'MoM Stats':
        fig = match_obj.plot_starPlayerStats()

    elif plot == 'MoM Events':
        fig = match_obj.plot_starPlayerKeyEvents()

    st.write(fig)

    b = io.BytesIO()
    fig.savefig(b, format='png', bbox_inches="tight")

    btn = st.download_button(
        label="Download Plot",
        data=b,
        file_name= plot.replace(' ', '') + "Plot.png",
        mime="image/png"
        )