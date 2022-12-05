##### LIMPIEDA DE DATOS #####

#importación librerias
import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("database_numbers.csv")
dataset.info()
dataset.head()

dataset.describe()
dataset.columns


#creación de la Clase
dataset["Class"]= np.where(dataset["Revenue"]>dataset["Budget"],1,0)

#mapas de correlacion
sns.heatmap(dataset[['Director_Name', 'Runtime', 'Genres', 'Movie_Title', 'Plot_Keywords',
       'Content_Rating', 'Budget', 'Aspect_Ratio', 'Movie_ID', 'Release_Date',
       'Revenue', 'Lead_Actor_ID', 'Lead_Actor_Name', 'Director_ID',
       'Studio_IDs', 'Studio_Names']].corr(), annot=True);

sns.heatmap(dataset[[
        'Genre_Musical', 'Genre_Film-Noir',
       'Genre_Romance', 'Genre_Sport', 'Genre_Music', 'Genre_Animation',
       'Genre_Adventure', 'Genre_News', 'Genre_Sci-Fi', 'Genre_Action',
       'Genre_History', 'Genre_Comedy', 'Genre_Horror', 'Revenue']].corr(), annot=True);

sns.heatmap(dataset[['Genre_Documentary', 'Genre_Mystery', 'Genre_Thriller', 'Genre_Family',
       'Genre_Drama', 'Genre_Crime', 'Genre_Fantasy',
       'Genre_War', 'Genre_Short', 'Genre_Western', 'Genre_Biography', 'Revenue']].corr(), annot=True);


sns.heatmap(dataset[[
       'Director_Avg_Movie_Revenue', 'Director_Movie_Count', 'Director_Ratio',
       'Keywords_Avg_Revenue', 'Keywords_Ratio', 'Content_Rating_Score',
       'Studios_Avg_Movie_Revenue', 'Studios_Ratio',
       'Lead_Actor_Avg_Movie_Revenue', 'Lead_Actor_Movie_Count',
       'Lead_Actor_Ratio', 'Revenue']].corr(), annot=True);


##comprobación NaN. E dataset está limpio

print(dataset['Director_Avg_Movie_Revenue'].isna().sum())
print(dataset['Keywords_Avg_Revenue'].isna().sum())
print(dataset['Studios_Avg_Movie_Revenue'].isna().sum())
print(dataset['Lead_Actor_Avg_Movie_Revenue'].isna().sum())
print(dataset['Budget'].isna().sum())


#comprobación de la distribución de las clases
100* dataset.Class.value_counts() / len(dataset.Class)

#plot
percentage=[42.35,57.64]

sns.set_palette("Paired")
ax=sns.countplot(x='Class', data=dataset)

patches = ax.patches
for i in range(len(patches)):
   x = patches[i].get_x() + patches[i].get_width()/2
   y = patches[i].get_height()+.05
   ax.annotate('{:.2f}%'.format(percentage[i]), (x, y), ha='center')

plt.show()


#nuevo dataframe solo con las variables que correlacionan con Class
database_final2= dataset[['Director_Avg_Movie_Revenue','Keywords_Avg_Revenue','Studios_Avg_Movie_Revenue','Lead_Actor_Avg_Movie_Revenue','Budget', 'Class']]
database_final2.head()

database_final2.to_csv("database_final2.csv")

###### GRIDSEARCHCV ######
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing
from sklearn import linear_model

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Metricas de clasificación
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,\
                            roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix


dataset = pd.read_csv("database_final2.csv")
dataset.head()

# seleccionamos las variables que correlacionan con Class
features = ['Director_Avg_Movie_Revenue', 'Studios_Avg_Movie_Revenue','Keywords_Avg_Revenue', 'Lead_Actor_Avg_Movie_Revenue', 'Budget']

# target para la y
target = pd.DataFrame(dataset.pop('Class'), columns=['Class'])

dataset = dataset[features]

X = dataset.values
y = np.ravel(target.values)

#selección hiperparámetros
# REGRESION LOGISTICA
grid_logreg = {                   
                     "penalty": ["l1","l2"], # Regularizaciones L1 y L2.
                     "C": [0.1, 0.5, 1.0, 5.0], # Cuanta regularizacion queremos
                     
                     "max_iter": [50,100,500],  # Iteraciones del Gradient Descent
                     
                     "solver": ["liblinear"]  
                    }


# KNN
grid_neighbors = {"n_neighbors": [3,5,7,9],       
                  "weights": ["uniform","distance"]  # Ponderar o no las clasificaciones en 
                                                     # función de la inversa de la distancia a cada vecino
                  }

# ARBOL DE DECISION
grid_arbol = {"max_depth":list(range(1,5)) # Profundidades del árbol. Cuanto más profundo, mas posibilidades de overfitting,
                                            # pero  mas preciso en entrenamiento.
              }

# RANDOM FOREST
grid_random_forest = {"n_estimators": [120], # El Random Forest no suele empeorar por exceso de
                                             # estimadores. A partir de cierto numero no merece la pena
                                             # perder el tiempo ya que no mejora mucho más la precisión.
                                             # Entre 100 y 200 es una buena cifra

                     
                     "max_depth": [3,4,5,6,10], # No le afecta tanto el overfitting como al decissiontree.
                                                      # Podemos probar mayores profundidades
                      
                     "max_features": ["sqrt", 3, 4] # Numero de features que utiliza en cada split.
                                                    # cuanto más bajo, mejor generalizará y menos overfitting.
                                                    
                     }
# SVM
grid_svm = {"C": [0.01, 0.1, 0.3, 0.5, 1.0, 3, 5.0, 15, 30], # Parametro de regularizacion
            "kernel": ["linear","rbf"], # Tipo de kernel, probar varios
            "gamma": [0.001, 0.1, "auto", 1.0, 10.0, 30.0] # Coeficiente de regulaizacion para los kernels
           }
           
# GRADIENT BOOSTING
grid_gradient_boosting = {"loss": ["deviance"], # Deviance suele ir mejor.
                          "learning_rate": [0.05, 0.1, 0.2, 0.4, 0.5],  # Cuanto más alto, mas aporta cada nuevo arbol
                          
                          "n_estimators": [20,50,100,200], # Cuidado con poner muchos estiamdores ya que vamos a
                                                           # sobreajustar el modelo
                          
                          "max_depth": [1,2,3,4,5], # No es necesario poner una profundiad muy alta. Cada nuevo
                                                    # arbol va corrigiendo el error de los anteriores.
                          
                          
                          "max_features": ["sqrt", 3, 4], # Igual que en el random forest
                          }


#pipelines

from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Si solo es el modelo, no hará falta meterlo en un pipeline

rand_forest = RandomForestClassifier()

svm = Pipeline([("scaler",MinMaxScaler()),
                ("selectkbest",SelectKBest()),
                ("svm",SVC())
               ])


reg_log = Pipeline([("imputer",SimpleImputer()),
                    ("scaler",MinMaxScaler()),
                    ("reglog",LogisticRegression())
                   ])


grid_random_forest = {"n_estimators": [120],
                     "max_depth": [3,4,5,6,10],
                     "max_features": ["sqrt", 3, 4]                          
                     }


svm_param = {                    
            'selectkbest__k': [1,2,3],
            'svm__C': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 
            'svm__kernel': ["linear","rbf"],
            'svm__coef0': [-10.,-1., 0., 0.1, 0.5, 1, 10, 100],
            'svm__gamma': ('scale', 'auto')
            }


reg_log_param = {    
                 "imputer__strategy": ['mean', 'median', 'most_frequent'],
                 "reglog__penalty": ["l1","l2"], 
                 "reglog__C": np.logspace(0, 4, 10)
                }


#división en train y test

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

# Almaceno en una lista de tuplas los modelos (nombre que le pongo, el modelo, hiperparametros)
models = [('rand_forest', rand_forest, grid_random_forest),
         ('svm', svm, svm_param),
         ('reg_log', reg_log, reg_log_param)]

# Declaro en un diccionario los pipelines e hiperparametros
models_gridsearch = {}

for i in models:
    models_gridsearch[i[0]] = GridSearchCV(i[1],
                                          i[2],
                                          cv=10,
                                          scoring="accuracy",
                                          verbose=1,
                                          n_jobs=-1)
    
    models_gridsearch[i[0]].fit(X_train, y_train)

#Resultado de los mejores modelos
best_grids = [(i, j.best_score_) for i, j in models_gridsearch.items()]

best_grids = pd.DataFrame(best_grids, columns=["Grid", "Best score"]).sort_values(by="Best score", ascending=False)
best_grids


models_gridsearch['rand_forest'].best_estimator_ #mejor estimador para random forest

models_gridsearch['rand_forest'].best_estimator_.score(X_test, y_test)


# Guardar el modelo
import pickle

with open('finished_model.model', "wb") as archivo_salida:
    pickle.dump(models_gridsearch['rand_forest'].best_estimator_, archivo_salida)


##### MODELO RANDOM FOREST ######
# Tratamiento de datos
# ------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Gráficos
# ------------------------------------------------------------------------------
import matplotlib.pyplot as plt

# Preprocesado y modelado
# ------------------------------------------------------------------------------
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz
from sklearn.tree import export_text
from sklearn.metrics import mean_squared_error


dataset = pd.read_csv("database_final2.csv")
dataset.head()

dataset = dataset [["Budget", "Director_Avg_Movie_Revenue", "Keywords_Avg_Revenue", "Studios_Avg_Movie_Revenue", "Lead_Actor_Avg_Movie_Revenue", "Class"]]
X = dataset.drop(["Class"],axis = 1)
y = dataset["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


#escalamos los datos

min_max_scaler = preprocessing.MinMaxScaler()

X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.transform(X_test)

# Para volver a leer el modelo
import pickle

with open('finished_model.model', "rb") as archivo_entrada:
    pipeline_importada = pickle.load(archivo_entrada)
    
print(pipeline_importada)

clf = RandomForestClassifier(bootstrap=True, criterion='gini',
            max_depth=3, max_features="sqrt", max_leaf_nodes=None,n_estimators=120)

clf.fit(X_train,y_train)


y_pred = clf.predict(X_test)

#report clasification
print(classification_report(y_test, y_pred))
c_mat = confusion_matrix(y_test,y_pred)
c_mat

#heatmap de la matriz de correlacion
sns.heatmap(c_mat/c_mat.sum(axis=1), annot=True);

#random forest con otros parámetros
clf_2 = RandomForestClassifier(bootstrap=True, criterion='gini',
            max_depth=10, max_features="sqrt", max_leaf_nodes=None,n_estimators=120)


clf_2.fit(X_train,y_train)


y_pred_2 = clf_2.predict(X_test)


print(classification_report(y_test, y_pred_2))
c_mat_2 = confusion_matrix(y_test,y_pred_2)
c_mat_2

sns.heatmap(c_mat_2/c_mat_2.sum(axis=1), annot=True);






