# "Más allá de Orion": Machine Learning para predecir la rentabilidad de una película

 ![comics-avengers-age-of-ultron-marvel-comics-ultron-wallpaper-preview](https://user-images.githubusercontent.com/98810076/205586753-16acfed8-2c05-4ca8-8228-35f38d211b93.jpg)

Este proyecto ha sido elaborado por César Herreros para el bootcamp de The Bridge en Madrid. En el mismo, se ha propuesto el objetivo de elaborar un algoritmo de clasificación que sea capaz de predecir si una película va a ser rentable o no.

La  base de datos ha sido obtenida en Kaggle.

Para determinar las features determinantes, he empleado mapas de calor para mostrar las correlaciones con la variable Revenue. Las variables que correlacionan positivamente son:

* Director Avg Movie REvenue: La media de recaudación de cada película en función del director
* Keywords_ Avg_REvenue: La media de recaudación de cada película en función de palabras clave sobre el argumento
* Studios_Avg_Movie_Revenue: La media de recaudación de cada película en función del estudio de producción
* Lead actor avg movie revenue:La media de recaudación de cada película en función del actor/atriz principal
* Budget: El presupuesto de la película

Para determinar si una película es rentable o no, se ha codificado una variable dummy:

* 0 -> La película no es rentable (Ingresos < presupuesto)
* 1 -> La película sí es rentable (Ingresos > presupuesto)


Para determinar el módelo óptimo que mejor se ajusta a mis datos, he aplicado GridSearchCV, siendo el resultado que el mejor modelo sería el Random Forest.

Este modelo ha conseguido una accuracy del 88%.

Los notebooks están separados en función del trabajo realizado en ellos: Uno para limpieza y procesado de datos, otro para el GridSearch y un tercero para ajustar el modelo Random Forest.

Librerías utilizadas: Numpy, Pandas, Matplotlib y Seaborn para los gráficos, sklearn para los procesos de machine learning. 
