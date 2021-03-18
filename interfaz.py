import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from lightfm.datasets import fetch_movielens
from lightfm import LightFM
from brewer2mpl import qualitative
from tabulate import tabulate
import cufflinks as cf
import plotly
import chart_studio.plotly as py
plotly.offline.init_notebook_mode()
cf.go_offline()
import warnings
warnings.filterwarnings('ignore')
import webbrowser
import os
#from utils import shapee
from tkinter import *
from tkinter import ttk
import calendar
from datetime import datetime
from math import ceil
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn import svm
import sklearn.manifold
import time

class PlotTsneRegion:
    def __init__(self, data, x_bounds, y_bounds, rand_points=None):
        self.data = data
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.rand_points = rand_points
        
    def get_slice(self):
        slice = self.data[
            (self.x_bounds[0] <= self.data.x) &
            (self.data.x <= self.x_bounds[1]) & 
            (self.y_bounds[0] <= self.data.y) &
            (self.data.y <= self.y_bounds[1])
        ]
        return slice
    
    def plot_region(self):
        slice = self.get_slice()
        if self.rand_points:
            slice = slice.sample(frac=self.rand_points)
        ax = slice.plot.scatter("x", "y", s=35, figsize=(10, 8))
        for i, point in slice.iterrows():
            ax.text(point.x + 0.02, point.y + 0.02, point.Name, fontsize=11)

def correlacion_pearson(video):
    start_time = time.time()    
    cols = ['Platform', 'Genre', 'Publisher', 'Developer', 'Rating']
    for col in cols:
        uniques = video[col].value_counts().keys()
        uniques_dict = {}
        ct = 0
        for i in uniques:
            uniques_dict[i] = ct
            ct += 1

        for k, v in uniques_dict.items():
            video.loc[video[col] == k, col] = v
    df1 = video[['Platform','Genre','Publisher','Year_of_Release','Critic_Score','Global_Sales']]
    df1 = df1.dropna().reset_index(drop=True)
    df1 = df1.astype('float64')
    mask = np.zeros_like(df1.corr())
    mask[np.triu_indices_from(mask)] = True
    cmap = sns.diverging_palette(730, 300, sep=20, as_cmap=True, s=85, l=15, n=20) # note: 680, 350/470
    with sns.axes_style("white"):
        fig, ax = plt.subplots(1,1, figsize=(15,8))
        ax = sns.heatmap(df1.corr(method='pearson'), mask=mask, vmax=0.2, square=True, annot=True, fmt=".2f", cmap=cmap)
        plt.yticks(rotation=0)
    plt.title('Correlacion de Pearson')
    print('Tiempo Correlacion de Pearson')
    print(time.time() - start_time)
    plt.show()


def correlacion_kendall(video):
    start_time = time.time()
    cols = ['Platform', 'Genre', 'Publisher', 'Developer', 'Rating']
    for col in cols:
        uniques = video[col].value_counts().keys()
        uniques_dict = {}
        ct = 0
        for i in uniques:
            uniques_dict[i] = ct
            ct += 1

        for k, v in uniques_dict.items():
            video.loc[video[col] == k, col] = v
    df1 = video[['Platform','Genre','Publisher','Year_of_Release','Critic_Score','Global_Sales']]
    df1 = df1.dropna().reset_index(drop=True)
    df1 = df1.astype('float64')
    mask = np.zeros_like(df1.corr())
    mask[np.triu_indices_from(mask)] = True
    cmap = sns.diverging_palette(730, 300, sep=20, as_cmap=True, s=85, l=15, n=20) # note: 680, 350/470
    with sns.axes_style("white"):
        fig, ax = plt.subplots(1,1, figsize=(15,8))
        ax = sns.heatmap(df1.corr(method='kendall'), mask=mask, vmax=0.2, square=True, annot=True, fmt=".2f", cmap=cmap)
        plt.yticks(rotation=0)
    plt.title('Correlacion de Kendall')
    print('Tiempo Correlacion de Kendall')
    print(time.time() - start_time)
    plt.show()

def correlacion_spearman(video):
    start_time = time.time()
    cols = ['Platform', 'Genre', 'Publisher', 'Developer', 'Rating']
    for col in cols:
        uniques = video[col].value_counts().keys()
        uniques_dict = {}
        ct = 0
        for i in uniques:
            uniques_dict[i] = ct
            ct += 1

        for k, v in uniques_dict.items():
            video.loc[video[col] == k, col] = v
    df1 = video[['Platform','Genre','Publisher','Year_of_Release','Critic_Score','Global_Sales']]
    df1 = df1.dropna().reset_index(drop=True)
    df1 = df1.astype('float64')
    mask = np.zeros_like(df1.corr())
    mask[np.triu_indices_from(mask)] = True
    cmap = sns.diverging_palette(730, 300, sep=20, as_cmap=True, s=85, l=15, n=20) # note: 680, 350/470
    with sns.axes_style("white"):
        fig, ax = plt.subplots(1,1, figsize=(15,8))
        ax = sns.heatmap(df1.corr(method='spearman'), mask=mask, vmax=0.2, square=True, annot=True, fmt=".2f", cmap=cmap)
        plt.yticks(rotation=0)
    plt.title('Correlacion de Spearman')
    print('Tiempo Correlacion de Spearman')
    print(time.time() - start_time)
    plt.show()

def critic_vs_global(video):
    cols = ['Platform', 'Genre', 'Publisher', 'Developer', 'Rating']
    for col in cols:
        uniques = video[col].value_counts().keys()
        uniques_dict = {}
        ct = 0
        for i in uniques:
            uniques_dict[i] = ct
            ct += 1

        for k, v in uniques_dict.items():
            video.loc[video[col] == k, col] = v
    df1 = video[['Platform','Genre','Publisher','Year_of_Release','Critic_Score','Global_Sales']]
    df1 = df1.dropna().reset_index(drop=True)
    df1 = df1.astype('float64')
    mask = np.zeros_like(df1.corr())
    mask[np.triu_indices_from(mask)] = True
    cmap = sns.diverging_palette(730, 300, sep=20, as_cmap=True, s=85, l=15, n=20) # note: 680, 350/470
    fig, ax = plt.subplots(1,1, figsize=(12,5))
    sns.regplot(x="Critic_Score", y="Global_Sales", data=df1.loc[df1.Year_of_Release >= 2014],
            truncate=True, x_bins=15, color="#75556c").set(ylim=(0, 2.5), xlim=(50, 90))
    plt.show()

def critic_vs_user_score(video):

    sns.jointplot('Critic_Score','User_Score',data=video,kind='hex', cmap= 'afmhot', size=11)
    plt.show()


def critic_score_vs_critic_count(video):

    sns.jointplot('Critic_Score','Critic_Count',data=video,kind='hex',cmap='afmhot', size=11)
    plt.show()

def cons_7g(video):
    video7th = video[(video['Platform'] == 'WiiU') | (video['Platform'] == 'PS3') | (video['Platform'] == 'X360')]
    #print(video7th.shape)
    plt.style.use('dark_background')
    yearlySales = video7th.groupby(['Year_of_Release','Platform']).Global_Sales.sum()
    yearlySales.unstack().plot(kind='bar',stacked=True, colormap= 'PuBu', grid=False,  figsize=(13,11))
    plt.title('Venta total de consolas 7ma generacion')
    plt.ylabel('Ventas globales')
    plt.show()

def venta_rating_7g(video):
    video7th = video[(video['Platform'] == 'WiiU') | (video['Platform'] == 'PS3') | (video['Platform'] == 'X360')]
    plt.style.use('dark_background')
    ratingSales = video7th.groupby(['Rating','Platform']).Global_Sales.sum()
    ratingSales.unstack().plot(kind='bar',stacked=True,  colormap= 'Greens', grid=False, figsize=(13,11))
    plt.title('Venta por Rating para consolas de 7MA generacion')
    plt.ylabel('Ventas')
    plt.show()

def venta_genero_7g(video):
    video7th = video[(video['Platform'] == 'WiiU') | (video['Platform'] == 'PS3') | (video['Platform'] == 'X360')]
    plt.style.use('dark_background')
    genreSales = video7th.groupby(['Genre','Platform']).Global_Sales.sum()
    genreSales.unstack().plot(kind='bar',stacked=True,  colormap= 'Reds', grid=False, figsize=(13,11))
    plt.title('Venta por generos 7ma G')
    plt.ylabel('Ventas')
    plt.show()

def ventas_usuarios(video):
    video7th = video[(video['Platform'] == 'WiiU') | (video['Platform'] == 'PS3') | (video['Platform'] == 'X360')]
    plt.style.use('seaborn-white')
    colors = ['#008DB8','#00AAAA','#00C69C']
    plt.figure(figsize=(15,11))
    plt.subplot(121)
    plt.pie(
   video7th.groupby('Platform').Global_Sales.sum(),
    labels=video7th.groupby('Platform').Global_Sales.sum().index,
    shadow=False,
    colors=colors,
    explode=(0.05, 0.05, 0.05),
    startangle=90,
    autopct='%1.1f%%'
    )
    plt.axis('equal')
    plt.title('Ventas globales 7ma G')
    plt.subplot(122)
    plt.pie(
    video7th.groupby('Platform').User_Count.sum(),
    labels=video7th.groupby('Platform').User_Count.sum().index,
    shadow=False,
    colors=colors,
    explode=(0.05, 0.05, 0.05),
    startangle=90,
    autopct='%1.1f%%'
    )
    plt.axis('equal')
    plt.title('Usuarios base 7ma G')
    plt.tight_layout()
    plt.show()

def juego_mas_popular_10(video):
    fig = video.sort_values('Global_Sales', ascending=False)[:10]
    fig = fig.pivot_table(index=['Name'], values=['NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales'], aggfunc=np.sum)
    fig = fig.sort_values('Global_Sales', ascending=True)
    fig = fig.drop('Global_Sales', axis=1)
    fig = fig.iplot(kind='barh', barmode='stack' , asFigure=True)
    fig.layout.margin.l = 350 
    fig.layout.xaxis.title='Ventas en millones de unidades'
    fig.layout.yaxis.title='Videojuegos'
    fig.layout.title = "Top 10 juegos mas populares"
    plotly.offline.plot(fig)
    plt.show(fig)

def venta_anual_genero(video):
    fig = (video.pivot_table(index=['Year_of_Release'], values=['Global_Sales'], columns=['Genre'], aggfunc=np.sum, dropna=False,)['Global_Sales']
        .iplot(subplots=True, subplot_titles=True, asFigure=True, fill=True,title='Ventas anuales por genero'))
    fig.layout.height= 800
    fig.layout.showlegend=False 
    plotly.offline.plot(fig)

def venta_anual_consolas_region(video):
    fig = video.pivot_table(index=['Platform'], values=['NA_Sales','EU_Sales','JP_Sales','Other_Sales'],                       aggfunc=np.sum, dropna=False,).iplot( asFigure=True,xTitle='Consola',yTitle='Ventas en millones',title='Venta de juegos por consola y por region')
    plotly.offline.plot(fig)

def hit(sales):
    if sales >= 1:
        return 1
    else:
        return 0

def definir_hits(dataset):
    dfb = dataset[['Name','Platform','Genre','Publisher','Year_of_Release','Critic_Score','Global_Sales']]
    dfb = dfb.dropna().reset_index(drop=True)
    df2 = dfb[['Platform','Genre','Publisher','Year_of_Release','Critic_Score','Global_Sales']]
    df2['Hit'] = df2['Global_Sales']
    df2.drop('Global_Sales', axis=1, inplace=True)
    df2['Hit'] = df2['Hit'].apply(lambda x: hit(x))

    n = ceil(0.05 * len(df2['Hit']))
    fig, ax = plt.subplots(1,1, figsize=(12,5))
    sns.regplot(x="Critic_Score", y="Hit", data=df2.sample(n=n), logistic=True, n_boot=500, y_jitter=.04, color="#75556c")
    plt.title('Analisis Critic score vs Global Sales')
    plt.show()

    df2[:5]
    df_copy = pd.get_dummies(df2)
    df3 = df_copy
    y = df3['Hit'].values
    df3 = df3.drop(['Hit'],axis=1)
    X = df3.values
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.50, random_state=2)

    radm = RandomForestClassifier(random_state=2).fit(Xtrain, ytrain)
    y_val_1 = radm.predict_proba(Xtest)
    print("Porcentaje RFC: ", sum(pd.DataFrame(y_val_1).idxmax(axis=1).values == ytest)/len(ytest))

    log_reg = LogisticRegression().fit(Xtrain, ytrain)
    y_val_2 = log_reg.predict_proba(Xtest)
    print("Porcentaje LR: ", sum(pd.DataFrame(y_val_2).idxmax(axis=1).values == ytest)/len(ytest))

    all_predictions = log_reg.predict(Xtest)
    print(classification_report(ytest, all_predictions))

    fig, ax = plt.subplots(figsize=(3.5,2.5))
    sns.heatmap(confusion_matrix(ytest, all_predictions), annot=True, linewidths=.5, ax=ax, fmt="d").set(xlabel='Valor predecido', ylabel='Valor esperado')
    plt.title('Matriz Confusion de Conj. Entrenamiento')
    plt.show()

    indices = np.argsort(radm.feature_importances_)[::-1]


    print('Ranking caracteristicas (top 20):')

    for f in range(10):
        print('%d. caracteristica: %d %s (%f)' % (f+1 , indices[f], df3.columns[indices[f]], radm.feature_importances_[indices[f]]))

    not_hit_copy = df_copy[df_copy['Hit'] == 0]
    df4 = not_hit_copy
    y = df4['Hit'].values
    df4 = df4.drop(['Hit'],axis=1)
    X = df4.values
    pred = log_reg.predict_proba(X)
    dfb = dfb[dfb['Global_Sales'] < 1]
    dfb['Hit_Probability'] = pred[:,1]
    dfb = dfb[dfb['Year_of_Release'] == 2016]
    dfb.sort_values(['Hit_Probability'], ascending=[False], inplace=True)
    dfb = dfb[['Name', 'Platform', 'Hit_Probability']]
    print("Top 20 Juegos con alta prob. de ser hits")
    print(dfb[:20].reset_index(drop=True))
    print("Top 20 Juegos con baja prob. de ser hits")
    print(dfb[:-21:-1].reset_index(drop=True))

def score_group(score):
    if score >= 90:
        return '90-100'
    elif score >= 80:
        return '80-89'
    elif score >= 70:
        return '70-79'
    elif score >= 60:
        return '60-69'
    elif score >= 50:
        return '50-59'
    else:
        return '0-49'

def in_top(x,pack):
    if x in pack:
        return x
    else:
        pass
def width(x):
    if x == 'Platform':
        return 14.4
    elif x == 'Developer':
        return 13.2
    elif x == 'Publisher':
        return 11.3
    elif x == 'Genre':
        return 13.6

def height(x):
    if x == 'Genre':
        return 8
    else:
        return 9

def analisis_critic_score(dataset):

    dfh = dataset.dropna(subset=['Critic_Score']).reset_index(drop=True)
    dfh['Score_Group'] = dfh['Critic_Score'].apply(lambda x: score_group(x))

    cols = ['Genre', 'Developer', 'Publisher', 'Platform']
    for col in cols:
        pack = []
        top = dfh[['Name', col]].groupby([col]).count().sort_values('Name', ascending=False).reset_index()[:15]
        for x in top[col]:
            pack.append(x)
        dfh[col] = dfh[col].apply(lambda x: in_top(x,pack))
        dfh_platform = dfh[[col, 'Score_Group', 'Global_Sales']].groupby([col, 'Score_Group']).median().reset_index().pivot(col,     "Score_Group", "Global_Sales")
        plt.figure(figsize=(width(col), height(col)))
        sns.heatmap(dfh_platform, annot=True, fmt=".2g", linewidths=.5).set_title((' \n'+col+' vs. critic score (por ventas medianas) \n'),     fontsize=18)
        plt.ylabel('',rotation='horizontal', fontsize=14)
        plt.xlabel('Grupo de puntuacion \n', fontsize=12)
        plt.yticks(rotation=0)
        pack = []
    plt.show()

def ohe_features_normalize_sales(data,cols):
    new_data = pd.get_dummies(data,columns=cols)
    new_data.dropna(inplace=True)
    new_data.reset_index(drop=True,inplace=True)
    new_data['Global_Sales'] = new_data['Global_Sales'] / new_data.groupby('Year_of_Release')['Global_Sales'].transform('sum')
    new_data['Year_of_Release'] = new_data['Year_of_Release'].astype(int)
    return new_data

def agrupar_datos(dataset):
    use_cols = ['Platform','Genre','Publisher','Developer','Rating']
    df_dummies = ohe_features_normalize_sales(dataset,use_cols)
    cols_to_use = list(df_dummies.columns)
    cols_to_use.remove('Name')
    matrix = df_dummies.as_matrix(columns=cols_to_use)
    tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)
    matrix_2d = tsne.fit_transform(matrix)
    df_tsne = pd.DataFrame(matrix_2d)
    df_tsne['Name'] = df_dummies['Name']
    df_tsne.columns = ['x','y', 'Name']
    cols = ['Name','x','y']
    df_tsne = df_tsne[cols]
    print(df_tsne.head(10))
    g = df_tsne.plot.scatter("x", "y", s=10, figsize=(20, 12), fontsize=20)
    g.set_ylabel('Y',size=20)
    g.set_xlabel('X',size=20)
    plt.show(g)

    x_bounds,y_bounds = (-52,-42),(61,71)
    region = PlotTsneRegion(df_tsne,x_bounds=x_bounds, y_bounds=y_bounds, rand_points=0.3)
    region.plot_region()
    plt.show(region)

    x_bounds,y_bounds = (0,50),(41,51)
    region = PlotTsneRegion(df_tsne,x_bounds=x_bounds, y_bounds=y_bounds, rand_points=0.3)
    region.plot_region()
    plt.show(region)

def primer_mapa_visual():
    webbrowser.open("primer_mapa_visual/index.html")

def segundo_mapa_visual():
    webbrowser.open("segundo_mapa_visual/index.html")

data = fetch_movielens(min_rating=4.0)
model = LightFM(loss='warp')
model.fit(data['train'], epochs=100, num_threads=4)
video = pd.read_csv('Video_Games_Sales_2016.csv')
#shapee()
dataset = video
dataset = dataset.copy()

root = Tk()
root.title("Prediccion de videojuegos Hits")

mainframe = ttk.Frame(root, padding="3 16 12 12")
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
mainframe.columnconfigure(0, weight=1)
mainframe.rowconfigure(0, weight=1)

feet = StringVar()
meters = StringVar()

feet_entry = ttk.Label(mainframe, text="Consolas 7G:", width=7)
feet_entry.grid(column=1, row=1, sticky=(W, E))

ttk.Button(mainframe, text="Total Ventas y usuarios", command=lambda: ventas_usuarios(video)).grid(column=2, row=2, sticky=W)
ttk.Button(mainframe, text="Venta consolas 7G", command=lambda: cons_7g(video)).grid(column=1, row=3, sticky=W)
ttk.Button(mainframe, text="Venta consolas 7G por Rating", command=lambda: venta_rating_7g(video)).grid(column=2, row=3, sticky=W)
ttk.Button(mainframe, text="Venta consolas 7G por Genero", command=lambda: venta_genero_7g(video)).grid(column=3, row=3, sticky=W)

feet_entry1 = ttk.Label(mainframe, text="Graficos Dinamicos:", width=7)
feet_entry1.grid(column=1, row=4, sticky=(W, E))
ttk.Button(mainframe, text="10 juegos mas populares", command=lambda: juego_mas_popular_10(video)).grid(column=1, row=5, sticky=W)
ttk.Button(mainframe, text="Ventas anuales por genero", command=lambda: venta_anual_genero(video)).grid(column=2, row=5, sticky=W)
ttk.Button(mainframe, text="Venta de juegos por consola y por region", command=lambda: venta_anual_consolas_region(video)).grid(column=3, row=5, sticky=W)

feet_entry2 = ttk.Label(mainframe, text="Analisis:", width=7)
feet_entry2.grid(column=1, row=6, sticky=(W, E))
ttk.Button(mainframe, text="Analisis Critic Score (4)", command=lambda: analisis_critic_score(dataset)).grid(column=2, row=7, sticky=W)
ttk.Button(mainframe, text="Critic Score vs Global Sales", command=lambda: critic_vs_global(video)).grid(column=1, row=8, sticky=W)
ttk.Button(mainframe, text="Critic vs User Score", command=lambda: critic_vs_user_score(video)).grid(column=2, row=8, sticky=W)
ttk.Button(mainframe, text="Critic Score vs Critic Count", command=lambda: critic_score_vs_critic_count(video)).grid(column=3, row=8, sticky=W)

feet_entry3 = ttk.Label(mainframe, text="Correlacion:", width=7)
feet_entry3.grid(column=1, row=9, sticky=(W, E))
ttk.Button(mainframe, text="Correlacion de Pearson", command=lambda: correlacion_pearson(video)).grid(column=1, row=10, sticky=W)
ttk.Button(mainframe, text="Correlacion de Kendall", command=lambda: correlacion_kendall(video)).grid(column=2, row=10, sticky=W)
ttk.Button(mainframe, text="Correlacion de Spearman", command=lambda: correlacion_spearman(video)).grid(column=3, row=10, sticky=W)

feet_entry3 = ttk.Label(mainframe, text="Clustering:", width=7)
feet_entry3.grid(column=1, row=11, sticky=(W, E))
ttk.Button(mainframe, text="Agrupar datos", command=lambda: agrupar_datos(dataset)).grid(column=2, row=12, sticky=W)

feet_entry4 = ttk.Label(mainframe, text="Prediccion:", width=7)
feet_entry4.grid(column=1, row=13, sticky=(W, E))
ttk.Button(mainframe, text="Definir Hits", command=lambda: definir_hits(dataset)).grid(column=2, row=14, sticky=W)

#feet_entry5 = ttk.Label(mainframe, text="Analisis visual:", width=7)
#feet_entry5.grid(column=1, row=15, sticky=(W, E))
#ttk.Button(mainframe, text="Agrupacion Jerarquica", command=lambda: primer_mapa_visual()).grid(column=2, row=16, sticky=W)
#ttk.Button(mainframe, text="arbol Ordenado Radial", command=lambda: segundo_mapa_visual()).grid(column=3, row=16, sticky=W)

for child in mainframe.winfo_children(): child.grid_configure(padx=5, pady=5)

root.bind('<Return>', correlacion_pearson)
root.mainloop()
