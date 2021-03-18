# Import the relevant libaries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from matplotlib import pyplot as plt
#matplotlib inline
from brewer2mpl import qualitative
from tabulate import tabulate
import cufflinks as cf
import plotly
import chart_studio.plotly as py
plotly.offline.init_notebook_mode()
cf.go_offline()
import warnings
warnings.filterwarnings('ignore')
import os


video = pd.read_csv('Video_Games_Sales_2016ok.csv')
print(video.head())
print("Tamanho Matriz")
print(video.shape)
print(video.isnull().any().any())
#Borramos los valores NULL
video = video.dropna(axis=0)


#Tabula los datos
tabulate(video.info(), headers='keys', tablefmt='psql')

#Imprimimos ls consolas
print(video.Platform.unique())

str_list = []
for colname, colvalue in video.iteritems():
    if type(colvalue[2]) == str:
         str_list.append(colname)

# Llegar a la columna numerica por inversion
num_list = video.columns.difference(str_list) 

# Crear la variable que contenga solo caracteristica numerica
video_num = video[num_list]
f, ax = plt.subplots(figsize=(14, 11))
plt.title('Correlacion de Pearson para caracteristicas numericas')
# Dibuja mpa de calor usando SeaBorn
sns.heatmap(video_num.astype(float).corr(method='pearson'),linewidths=0.25,vmax=1.0, 
            square=False, cmap="cubehelix_r", linecolor='k', annot=True)
plt.show()

#####Crtic_Score vs User_Score###########
video['User_Score'] = video['User_Score'].convert_objects(convert_numeric= True)
sns.jointplot(x='Critic_Score',y='User_Score',data=video,kind='hex', cmap= 'afmhot', size=11)

plt.show()

#####Crtic_Score vs Critic_Count###########
sns.jointplot('Critic_Score','Critic_Count',data=video,kind='hex',cmap='afmhot', size=11)
plt.show()

########## Consolas 7MA generacion #################################
video7th = video[(video['Platform'] == 'WiiU') | (video['Platform'] == 'PS3') | (video['Platform'] == 'X360')]
print(video7th.shape)

### plot venta total de consolas 7ma generacion ########
plt.style.use('dark_background')
yearlySales = video7th.groupby(['Year_of_Release','Platform']).Global_Sales.sum()
yearlySales.unstack().plot(kind='bar',stacked=True, colormap= 'PuBu',  
                           grid=False,  figsize=(13,11))
plt.title('Venta total de consolas 7ma generacion')
plt.ylabel('Global Sales')
plt.show()


############ plot Venta de video juegos para consola, por Rating ######################
plt.style.use('dark_background')
ratingSales = video7th.groupby(['Rating','Platform']).Global_Sales.sum()
ratingSales.unstack().plot(kind='bar',stacked=True,  colormap= 'Greens', 
                           grid=False, figsize=(13,11))
plt.title('Venta por Rating para consolas de 7MA generacion')
plt.ylabel('Sales')
plt.show()

########################venta por generos para cada consola ######################
plt.style.use('dark_background')
genreSales = video7th.groupby(['Genre','Platform']).Global_Sales.sum()
genreSales.unstack().plot(kind='bar',stacked=True,  colormap= 'Reds', 
                          grid=False, figsize=(13,11))
plt.title('Venta por generos')
plt.ylabel('Sales')
plt.show()

############################   Total de ventas y Total de usurios    ###############

# Plotting our pie charts
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
plt.title('Ventas globales')

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
plt.title('Usurios base')
plt.tight_layout()
plt.show()



##############visualizacion dinamica#######################
fig = video.sort_values('Global_Sales', ascending=False)[:10]
fig = fig.pivot_table(index=['Name'], values=['NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales'], aggfunc=np.sum)
fig = fig.sort_values('Global_Sales', ascending=True)
fig = fig.drop('Global_Sales', axis=1)
fig = fig.iplot(kind='barh', barmode='stack' , asFigure=True) #For plotting with Plotly
fig.layout.margin.l = 350 #left margin distance
fig.layout.xaxis.title='Sales in Million Units'# For setting x label
fig.layout.yaxis.title='Title' # For setting Y label
fig.layout.title = "Top 10 global game sales" # For setting the graph title
plotly.offline.iplot(fig) # Show the graph
plt.show(fig)
