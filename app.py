"""
Spotify Hit vs Flop Explorer

An interactive Dash app that visualizes the likelihood of a song becoming a hit based on selected audio features.
Users can explore feature distributions, model predictions, and positional trends using logistic regression and multiple interactive plots.

Created: April 2025
Author: Julie Castro Pena
"""
import dash
from dash import html, dcc, Input, Output
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import statsmodels.api as sm

# Setting default Plotly theme
pio.templates.default = "plotly_white"


# Loading the data
spotify_data = pd.read_csv('Spotify_dataset.csv')

# Data Cleaning
spotify_data.drop(['Unnamed: 0', 'genres'], axis=1, inplace=True)
spotify_data.drop_duplicates(subset=['uri'], inplace=True)

# Defining columns of interest for analysis
columns_of_interest = ['danceability', 'energy', 'valence', 'instrumentalness', 'speechiness']


# Logistic regression setup
X =  sm.add_constant(spotify_data[columns_of_interest])
y = spotify_data['target']
model = sm.Logit(y,X).fit(disp=False)
coefs = model.params  # Extracting model coefficients
conf = model.bse * 1.96 # Computing 95% confidence intervals

# Initializing the Dash app
app = dash.Dash(__name__)
app.title = "Spotify Hit Predictor"

# Structuring the layout with Markdown, dropdown, and multiple graphs
app.layout = html.Div([
    html.H1("Spotify Hit vs Flop Explorer", style={"textAlign": "center"}),

    # Intro text describing the app
    dcc.Markdown("""
    **Welcome to the Spotify Hit vs Flop Explorer**

    This interactive dashboard examines the characteristics that influence whether a song becomes a hit or remains under the radar.  
    Explore how musical features like *danceability*, *energy*, and *valence* relate to a track’s success, based on data from Spotify’s Web API and Billboard charts.
    """, style={"backgroundColor": "#fff8f5", "padding": "20px", "borderRadius": "12px"}),

    # Dropdown menu to explore individual features
    html.Label("Explore how each feature is distributed across hits and flops:"),
    dcc.Dropdown(
        id='feature-dropdown',
        options=[{'label': col.capitalize(), 'value': col} for col in columns_of_interest],
        value='danceability',
        clearable=False
    ),

    html.Div(dcc.Graph(id='histogram'), style={"maxWidth": "900px", "margin": "0 auto", "padding": "20px"}),

    html.H2("Feature Correlation with Hit Likelihood", style={"marginTop": "40px"}),

     # Explaining correlation logic
    dcc.Markdown("""
    These values reflect how each feature correlates with a track being a hit.

    - Positive correlation → feature is more common in hits  
    - Negative correlation → feature is more common in flops  
    """, style={"backgroundColor": "#f5faff", "padding": "20px", "borderRadius": "12px"}),

    html.Div(dcc.Graph(id='correlation-bar'), style={"maxWidth": "900px", "margin": "0 auto", "padding": "20px"}),

    html.H2("Predictive Strength of Each Feature (Logistic Regression Model)", style={"marginTop": "40px"}),

    dcc.Markdown("""
    This plot shows which features are statistically significant in predicting hit potential.  
    Error bars represent 95% confidence intervals.
    """, style={"backgroundColor": "#f5faff", "padding": "20px", "borderRadius": "12px"}),

    html.Div(dcc.Graph(id='regression-bar'), style={"maxWidth": "900px", "margin": "0 auto", "padding": "20px"}),

    html.H2("Check a Song's Hit Probability", style={"marginTop": "40px"}),

    dcc.Markdown("""
    Type in the **artist** and **track name** to see if the song exists in the database and what the model predicts about its hit potential.

    You’ll see your song highlighted with a ★ purple star.
    """, style={"backgroundColor": "#fff8f5", "padding": "20px", "borderRadius": "12px"}),

    html.Div([
        html.Div([
            html.Label("Artist Name:"),
            dcc.Input(id='artist-input', type='text', placeholder='e.g. Ariana Grande', style={'width': '100%'})
        ], style={"marginBottom": "10px"}),

        html.Div([
            html.Label("Track Name:"),
            dcc.Input(id='track-input', type='text', placeholder='e.g. eternal sunshine', style={'width': '100%'})
        ], style={"marginBottom": "10px"}),

        html.Button('Predict', id='predict-button', n_clicks=0, style={"marginBottom": "20px"}),

        html.Div(id='prediction-output', style={
            "padding": "15px", "backgroundColor": "#f0f0f0", "borderRadius": "10px",
            "fontsize": "16px", "textAlign": "center"
        })
    ], style={"maxWidth": "600px", "margin": "0 auto"}),

    html.H2("Attribute Spread by Outcome", style={"marginTop": "40px"}),

    dcc.Markdown("""
    This jitter plot shows how your selected track compares to **all hits and flops** on the chosen audio feature.  
    You will see your song highlighted with a ★ purple star.

    Use it to explore how features like *energy*, *valence*, or *instrumentalness* vary — and where your track fits in the spread.
    """, style={"backgroundColor": "#f5faff", "padding": "20px", "borderRadius": "12px"}),

    html.Div(dcc.Graph(id='jitter-plot'), style={"maxWidth": "900px", "margin": "0 auto", "padding": "20px"}),

    html.H2("Song Placement on Feature Space", style={"marginTop": "40px"}),

    dcc.Markdown("""
    This scatter plot shows where your selected track falls in the space defined by two of the most predictive features: Danceability and Energy.

    It gives you a visual sense of why the model might predict a song as a hit or a flop.
    """, style={"backgroundColor": "#fff8f5", "padding": "20px", "borderRadius": "12px"}),

    html.Div(dcc.Graph(id='scatter-song-placement'), style={"maxWidth": "900px", "margin": "0 auto", "padding": "20px"})
])




# Callback to update histogram
# Creating a grouped histogram that compares the selected audio feature across hit and flop tracks
@app.callback(
    Output('histogram', 'figure'),
    Input('feature-dropdown', 'value')
)
def update_histogram(selected_feature):
    """Updating histogram comparing selected feature between hits and flops."""
    fig = px.histogram(
        spotify_data,
        x=selected_feature,
        color='target',
        nbins=30,
        barmode='group',
        opacity=0.85,
        color_discrete_map={0: '#F7A8A4', 1: '#A8E6CF'},
        labels={'target': 'Track Type', selected_feature: selected_feature.capitalize()},
        title=f"Distribution of {selected_feature.capitalize()} by Hit/Flop",
        template='plotly_white'
    )
    fig.update_layout(
        legend_title_text='Track Type',
        legend=dict(
            itemsizing='constant',
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font=dict(
            family="Arial Rounded MT Bold, sans-serif",
            size=14,
            color="#444"
        )
    )
    return fig

# Callback to update correlation bar
# Calculating and displaying the correlation of each feature with the target (hit vs flop)
@app.callback(
    Output('correlation-bar', 'figure'),
    Input('feature-dropdown', 'value')  
)
def update_correlation(selected_feature):
    """Display correlation between each feature and hit likelihood."""
    features = ['danceability', 'energy', 'valence', 'instrumentalness', 'speechiness']
    corr = spotify_data[columns_of_interest + ['target']].dropna().corr()['target'].drop('target')

    # Determining the bar colors: highlighting selected and fading the others
    bar_colors = [
        '#A8E6CF' if val >= 0 else '#F7A8A4'
        for val in corr.values
    ]
    fig = px.bar(
        x=corr.index,
        y=corr.values,
        text=[f"{val:.2f}" for val in corr.values],
        labels={'x': 'Feature', 'y': 'Correlation'},
        title="How Each Feature Correlates with Track Hit Status.",
    )

    fig.update_traces(
        marker_line_color='white',
        marker_color=bar_colors, 
        textposition='outside')
    
    fig.update_layout(
        yaxis=dict(title='Correlation with Target',
        range=[
            min(-0.5, corr.min() - 0.05),
            max(0.5, corr.max() + 0.1)
        ]),
        xaxis=dict(title='Audio Feature'),
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font=dict(
            family="Arial Rounded MT Bold, sans-serif",
            size=14,
            color="#444"
        ),
        coloraxis_showscale=False
    )
    return fig

# Callback to update regression-bar

@app.callback(Output('regression-bar', 'figure'), Input('feature-dropdown', 'value'))
def update_regression(_):
    """Show logistic regression coefficients with confidence intervals."""
    fig = go.Figure()
    fig. add_trace(go.Bar(
        x=coefs.index,
        y=coefs.values,
        error_y=dict(type='data', array=conf.values, visible=True),
        marker_color=['#A8E6CF' if coefs[c] >= 0 else '#F7A8A4' for c in coefs.index],
        textposition='outside'
    ))
    fig.update_layout(
        title="Logistic Regression Coefficients (with 95% CI)",
        yaxis_title="Coefficient Value",
        xaxis_title="Feature",
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font=dict(family="Arial Rounded MT Bold, sans-serif", size=14, color="#444")
    )
    return fig

# Callback to update jitter plot
@app.callback(
    Output('jitter-plot', 'figure'), 
    Input('feature-dropdown','value'),
    Input('predict-button', 'n_clicks'),
    Input('artist-input', 'value'),
    Input('track-input', 'value')
)

def update_jitter(selected_feature, n_clicks, artist_name, track_name):
    """Render jitter plot and highlight a selected song if found."""
    # Base jitter plot of all songs
    fig = px.strip(
        spotify_data,
        x='target',
        y=selected_feature,
        color='target',
        stripmode='overlay',
        color_discrete_map={0: '#F7A8A4', 1: '#A8E6CF'},
        labels={'target': 'Track Type', selected_feature: selected_feature.capitalize()},
        title=f"Jitter Plot of {selected_feature.capitalize()} by Track Type"
    )

    # Adding a marker for the selected song, if valid input is provided
    if n_clicks > 0 and artist_name and track_name:
        track = spotify_data[
            (spotify_data['artist'].str.lower() == artist_name.strip().lower()) &
            (spotify_data['track'].str.lower() == track_name.strip().lower())
        ]
        if not track.empty and not pd.isna(track[selected_feature].iloc[0]):
            fig.add_trace(
                go.Scatter(
                    x=[track['target'].iloc[0]],
                    y=[track[selected_feature].iloc[0]],
                    mode='markers+text',
                    name='Selected Song',
                    text=[f"{track['track'].iloc[0]} by {track['artist'].iloc[0]}"],
                    textposition='top center',
                    marker=dict(
                        size=16, 
                        color='purple',
                        symbol='star',
                        line=dict(width=2, color='black')
                    ),
                    showlegend=True,
                    legendgroup='selected',
                    hoverinfo='text'
    
                )
            )

    fig.update_layout(
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font=dict(family="Arial Rounded MT Bold, sans-serif", size=14, color="#444")
    
    )
    return fig

# Callback to update prediction-output
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    Input('artist-input', 'value'),
    Input('track-input', 'value')
    )
def predict_hit(n_clicks, artist_name, track_name):
     """Return hit/flop prediction and probability for the input song."""
    if n_clicks == 0 or not artist_name or not track_name:
        return ""
    
    track = spotify_data[
        (spotify_data['artist'].str.lower() == artist_name.strip().lower()) &
        (spotify_data['track'].str.lower() == track_name.strip().lower())
    ]

    if track.empty:
        return "Song not found in the dataset. Try another one!"
    
    track_features = track[columns_of_interest]
    track_features_const = sm.add_constant(track_features, has_constant='add')

    probability = model.predict(track_features_const)[0]
    result = "**Hit**" if probability >= 0.5 else "**Flop**"

    return html.Div([
        html.P(f"Prediction: {result}", style={"fontWeight": "bold", "fontSize": "18px"}),
        html.P(f"Probability of being a hit: {probability: .2%}")
    ])

# Callback to update scatter song placement
@app.callback(
    Output('scatter-song-placement', 'figure'),
    Input('predict-button', 'n_clicks'),
    Input('artist-input', 'value'),
    Input('track-input', 'value')
)
def update_song_scatter(n_clicks, artist_name, track_name):
    """Display selected song's position on Danceability vs Energy plot."""
    # Base figure using graph_objects for full control
    fig = go.Figure()

    # Adding scatter points for all songs by target group
    for target_value, color in zip([0, 1], ['#F7A8A4', '#A8E6CF']):
        filtered = spotify_data[spotify_data['target'] == target_value]
        fig.add_trace(go.Scatter(
            x=filtered['danceability'],
            y=filtered['energy'],
            mode='markers',
            name=f"{'Flop' if target_value == 0 else 'Hit'}",
            marker=dict(color=color, size=5),
            opacity=0.5,
            hoverinfo='skip'
        ))

    # Adding highlighted selected song, if found
    if n_clicks > 0 and artist_name and track_name:
        track = spotify_data[
            (spotify_data['artist'].str.lower() == artist_name.strip().lower()) &
            (spotify_data['track'].str.lower() == track_name.strip().lower())
        ]
        if not track.empty:
            fig.add_trace(go.Scatter(
                x=[track['danceability'].iloc[0]],
                y=[track['energy'].iloc[0]],
                mode='markers+text',
                name='Selected Song',
                text=[f"{track['track'].iloc[0]} by {track['artist'].iloc[0]}"],
                textposition='top center',
                marker=dict(
                    size=16,
                    color='purple',
                    symbol='star',
                    line=dict(width=2, color='black')
                ),
                hoverinfo='text',
                showlegend=True,
                legendgroup='selected'
            ))

    fig.update_layout(
        title="Danceability vs Energy (with Your Song Highlighted)",
        xaxis_title='Danceability',
        yaxis_title='Energy',
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font=dict(family="Arial Rounded MT Bold, sans-serif", size=14, color="#444")
    )

    return fig



# Running the app
if __name__ == '__main__':
    app.run(debug=True)
