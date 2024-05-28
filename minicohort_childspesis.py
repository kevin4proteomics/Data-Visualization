import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import dash
from dash import Dash, html,dcc, Input, Output, State, dash_table
import dash_bio as dashbio
import dash_daq as daq
import mpld3
import dash_bootstrap_components as dbc
import plotly.express as px
from plotly.figure_factory import create_dendrogram
import plotly.figure_factory as ff
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate
import os
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
#define h clustering method

def plot_hclustering(df_scaled):
    # Perform hierarchical clustering
    linked = linkage(df_scaled, 'single')

    # Create dendrogram
    plt.figure(figsize=(10, 7))
    dendrogram(linked,
               orientation='top',
               labels=range(1, 11),
               distance_sort='descending',
               show_leaf_counts=True)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.show()

    # Plot clustering heatmap
    sns.clustermap(df_scaled, method='single', cmap='viridis', standard_scale=1)
    plt.title('Clustering Heat')


#load dataframe
df = pd.read_csv('https://ftp.pride.ebi.ac.uk/pride/data/archive/2024/02/PXD049377/NY1313_1326_1351_Protein_Report.csv')
for c in df.columns:
    print(c)
paper_url = 'https://journals.physiology.org/doi/full/10.1152/ajplung.00164.2021'
project_directory = '/Users/yen-yuyang/Dropbox/Kevin/DataAnalysis/python_scripts/BiomarkerMiniProject/'

#change quantity to float data type
cols = df.columns[df.columns.str.contains('Quantity')]
for col in cols:
    df[col] = df[col].replace('Filtered', np.nan)
    df[col] = pd.to_numeric(df[col])

A_cols = [col for col in df.columns if "DIA_A" in col or "NY1351_A" in col]
S_cols = [col for col in df.columns if "DIA_S" in col or "NY1351_S" in col]
# assuming df_protein is your DataFrame
DIA_cols = df[df.columns[df.columns.str.contains('PG.Quantity')]].columns.tolist()

# Apply log2 transformation
df[DIA_cols] = df[DIA_cols].applymap(np.log2)
df_mean = df.copy()
# df_outliers = pd.DataFrame()
df_mean['average_A'] = df[A_cols].mean(axis=1)
df_mean['average_S'] = df[S_cols].mean(axis=1)

#compute ratio and p values between S adn A samples
ratios = []
p_values = []
A_cols = [col for col in df.columns if "DIA_A" in col or "NY1351_A" in col]
S_cols = [col for col in df.columns if "DIA_S" in col or "NY1351_S" in col]
DIA_S_cols = df[S_cols]
DIA_A_cols = df[A_cols]

for i in range(len(DIA_A_cols)):
    # Find ratio of means
    ratio = DIA_S_cols.iloc[i, :].mean() - DIA_A_cols.iloc[i, :].mean()
    ratios.append(ratio)

    # Perform t-test
    ttest_result = stats.ttest_rel(DIA_S_cols.iloc[i, :], DIA_A_cols.iloc[i, :], nan_policy='omit')
    p_values.append(ttest_result.pvalue)

# Add results to a new dataframe
df_volcano = df[["PG.Genes", "PG.ProteinAccessions", "PG.ProteinNames","PG.ProteinDescriptions"]]
df_volcano.rename(columns={"PG.Genes": "Protein_Group",
                            "PG.ProteinAccessions": "Accession_Number",
                           "PG.ProteinNames": "Protein_Full_Name",
                            "PG.ProteinDescriptions":"Protein_Description"
                           }, inplace=True)

df_volcano['EFFECTSIZE'] = pd.Series(ratios)
df_volcano['P'] = pd.Series(p_values)
df_volcano['neg p_value'] = pd.Series(-np.log10(p_values))
df_volcano.dropna(subset=["P"], inplace=True)
df_volcano.to_csv(os.path.join(project_directory,'df_trimmed.csv'))
empty_rows = df_volcano[df_volcano.isnull().any(axis=1)]

app=Dash(__name__,
         external_stylesheets=[dbc.themes.SOLAR])
df_volcano.to_csv(os.path.join(project_directory,'df_trimmed.csv'))
df_volcano = pd.read_csv(os.path.join(project_directory,'df_trimmed.csv'))


#DASH
app.layout = html.Div([
    html.H3('Effect sizes'),
    dcc.RangeSlider(
        id='default-volcanoplot-input',
        min=-5,
        max=5,
        step=0.5,
        marks={i: {'label': str(i)} for i in range(-5, 5)},
        value=[-1,1]
    ),
    html.Br(),
    dbc.Row(
        [
        dbc.Col(
            html.Div(
            children= [
                html.H4(children='Volcano plot 1'),
                dcc.Graph(id='dashbio-default-volcanoplot')
                ]
        ),
        width=4),
        dbc.Col(
            html.Div(
                children=[
                    html.H4(children='Volcano plot 2'),
                    dcc.Graph(id='dashbio-default-volcanoplot2')
                ],
            ),
        width=4),
        dbc.Col(
            html.Div(
            children= [
                html.H4(children='Volcano plot 3'),
                dcc.Graph(id='dashbio-default-volcanoplot3')
                ]

        ),
        width=4)

        ]

    ),
    html.Br(),
    html.H3("Classification Analysis"),
    html.Div(
             children=[
                dcc.Dropdown(id='pca-n',
                             options=[
                                {'label': str(i), 'value': i} for i in range(1, 11)
                                # {'label':2, 'value':2},
                                # {'label':3, 'value':3},
                                # {'label':4, 'value':4},
                                # {'label':5, 'value':5},
                                # {'label':6, 'value':6}
                             ],
                             value=2

                             ),
                dcc.Dropdown(id='pca_scale',
                             options=[
                                 {'label':'Scale', 'value':1},
                                 {'label':'Do not scale', 'value':2}
                             ],
                             value=1),
                dcc.Dropdown(id='pca_imputation',
                                             options=[
                                                 {'label':'Remove missing value', 'value':1},
                                                 {'label':'Simple mean imputation', 'value':2},
                                                 {'label':'Simple median imputation', 'value':3},
                                                 {'label':'KNN imputation', 'value':4}
                                             ],
                                             value=1),

                 html.Br(),
                 html.H3("PC Starting from: "),
                 dcc.Input(id='pca_start', value=1),
                 dcc.Input(id='knn_neighbor',
                           value=3)
             ]),
    html.Br(),
    html.Div(
        dcc.Dropdown(id='hclustering-method',
                     options=[{
                        'label':'Remove missing value', 'value':1
                     }])
    ),

    dbc.Row(
        [
            dbc.Col(
                html.Div(
                    children=[
                        html.H4(children='PCA'),
                        html.Div(id='pca-plot')
                    ]),
                width=4
            ),
            dbc.Col(
                html.Div(
                    children=[
                        html.H4(children='Hierarchical Clustering'),
                        html.Div(id='hclustering-plot')]
                ),
                width=4
            )
        ]),

    html.Br()

])

@app.callback(
    [Output('dashbio-default-volcanoplot', 'figure'),
        Output('dashbio-default-volcanoplot2', 'figure'),
     Output('dashbio-default-volcanoplot3', 'figure')],
    Input('default-volcanoplot-input', 'value')
)
def update_volcanoplot(effects):
    fig_volcano = dashbio.VolcanoPlot(
        dataframe=df_volcano,
        snp="Protein_Description",
        gene="Protein_Group",
        genomewideline_value=-np.log10(0.05),
        effect_size_line=effects,
        effect_size_line_color='#AB63FA',
        effect_size_line_width=2,
        genomewideline_color='#EF553B',
        genomewideline_width=2,
        highlight_color='#119DFF',
        col='#2A3F5F',
        point_size = 10,
    )

    fig_volcano2 = dashbio.VolcanoPlot(
        dataframe=df_volcano,
        snp="Protein_Description",
        gene="Protein_Group",
        annotation="Protein_Group",
        genomewideline_value=-np.log10(0.05),
        effect_size_line=effects,
        effect_size_line_color='#AB63FA',
        effect_size_line_width=2,
        genomewideline_color='#EF553B',
        genomewideline_width=2,
        highlight_color='#119DFF',
        col='#2A3F5F',
        point_size = 10,
    )
    fig_volcano3 = dashbio.VolcanoPlot(
        dataframe=df_volcano,
        snp="Protein_Description",
        gene="Protein_Group",
        genomewideline_value=-np.log10(0.05),
        effect_size_line=effects,
        effect_size_line_color='#AB63FA',
        effect_size_line_width=2,
        genomewideline_color='#EF553B',
        genomewideline_width=2,
        highlight_color='#119DFF',
        col='#2A3F5F',
        point_size = 10,)

    return fig_volcano, fig_volcano2, fig_volcano3

# @app.callback(
#     [Output('pca_start', 'options')],
#     [Input('pca-n', 'value')]
#
# )
#
# def pca_start_update(pca_n):
#     if pca_n <= 3:
#         pca_start_dict = [{'label': 'PC1', 'value':1},{'label': 'PC2', 'value':2}]
#     elif pca_n > 3:
#         pca_start_dict = [{'label': f'PC-{str(i)}', 'value': i} for i in range(1, pca_n)]
#         print(pca_start_dict)
#     options = pca_start_dict
#     return options

@app.callback(
    [Output('pca-plot', 'children'),
     Output('hclustering-plot','children')],

    [Input('pca-n', 'value'),
     Input('pca_scale','value'),
     Input('pca_imputation','value'),
     Input('knn_neighbor','value'),
     Input('pca_start','value')]
#Output('hclustering-plot', 'figure')
)

def update_pca(pca_n, pca_scale, pca_imputation, knn_neighbor, pca_start):
    # PCA plot
    # pg_quantity_cols = [col for col in df.columns if 'PG.Quantity' in col]

    if knn_neighbor:
        knn_neighbor= int(knn_neighbor)
    df_pca = df[S_cols + A_cols]

    #decide to remove or impute missing data
    if pca_imputation == 1:
        df_pca.dropna(inplace=True)
    elif pca_imputation != 1:
        if pca_imputation == 2:
            imputer = SimpleImputer(strategy='mean')
        elif pca_imputation == 3:
            imputer = SimpleImputer(strategy='median')
        elif pca_imputation == 4:
            imputer = KNNImputer(n_neighbors=knn_neighbor)
        df_pca_imputed = imputer.fit_transform(df_pca)
        df_pca = pd.DataFrame(df_pca_imputed, columns=df_pca.columns)

    df_pca = df_pca.T
    print('PCA: ', df_pca)

    if pca_scale == 1:
        scaler = StandardScaler()  # standardize data
        df_pca_scaled = scaler.fit_transform(df_pca)
    elif pca_scale == 2:
        df_pca_scaled = df_pca.copy()

    pca = PCA(n_components=pca_n)
    df_pca_fitted = pca.fit(df_pca_scaled)
    df_pca_results = pca.transform(df_pca_scaled)
    # Create a DataFrame for the PCA results
    pca_color = ['S'] * 9 + ['A'] * 9

    #clustering
    # plot_hclustering(df_scaled=df_pca)

    if pca_n == 2:
        pca_df = pd.DataFrame(data=df_pca_results, columns=['PC1', 'PC2'])
        pca_df['Color'] = pca_color
        fig_pca = px.scatter(pca_df, x='PC1', y='PC2', color='Color')
    elif pca_n !=2:
        columns =[]
        for i in range(pca_n):
            pc = 'PC' + str(pca_n-i)
            columns.append(pc)
        columns = sorted(columns, reverse=False)
        pca_df = pd.DataFrame(data=df_pca_results, columns=columns)
        pca_df['Color'] = pca_color
        pca_start = int(pca_start)-1
        fig_pca = px.scatter_3d(pca_df, x=columns[pca_start], y=columns[pca_start+1], z=columns[pca_start+2], color='Color')

    print('PCA explained:')
    print(df_pca_fitted.explained_variance_ratio_)



    # # #create dendrogram
    # df_pca_scaled = pd.DataFrame(df_pca_scaled)
    # df_pca_scaled = df_pca_scaled.T
    # df_pca_scaled.columns = df_pca.index.tolist()
    # df_pca_scaled.index = df_pca.columns.tolist()
    # print('df_pca_scaled: ', df_pca_scaled)
    # hclustering_plot = plot_hclustering(df_pca_scaled.T)
    df_clustering = pd.DataFrame(df_pca_scaled)
    df_clustering['Label'] = df_pca.index
    df_clustering.set_index('Label', inplace=True)

    print('clustering df', df_clustering)
    linkage_matrix = linkage(df_clustering, method='single')
    def custom_linkage_func(*args):
        return linkage_matrix
    fig_cluster = ff.create_dendrogram(df_clustering,
                                       orientation='bottom',
                                       labels=df_clustering .index,
                                       linkagefun=custom_linkage_func)
    fig_cluster.update_layout(width=600, height=450, title='Dendrogram with Plotly')

    return (dcc.Graph(figure=fig_pca), dcc.Graph(figure=fig_cluster))


if __name__ == '__main__':
    app.run(debug=True)