## Overfitting and model tunning

# Clean data and load packages
source("get_and_clean_data.R")
library(caret)



# Create training and cross-validation data sets directly
# Use objective variable to distribute cv points
set.seed(22)
training.rows <- createDataPartition(y = all.training.data$Sales,
                                     p = 0.8,
                                     list = FALSE)
training.set <- all.training.data[training.rows, ]
cv.set <- all.training.data[-training.rows, ]



# Create training and cross-validation data sets using resampling
# Use objective variable to distribute cv points
# Multiple splits
set.seed(22)
repeated.splits <- createDataPartition(y = all.training.data$Sales, p = 0.8,
                                       times = 3)
# And create sets in the same way



# K folds cross validation
set.seed(22)
training.splits <- createFolds(y = all.training.data$Sales, k = 10,
                               returnTrain = TRUE)
training.set.1 <- all.training.data[training.splits[[1]], ]
cv.set.1 <- all.training.data[-training.splits[[1]], ]
training.set.2 <- all.training.data[training.splits[[2]], ]
cv.set.2 <- all.training.data[-training.splits[[2]], ]
training.set.3 <- all.training.data[training.splits[[3]], ]
cv.set.3 <- all.training.data[-training.splits[[3]], ]
# ...



# Bootstrapping (training set del tamaño de los datos disponibles, 
# cogiendo puntos con reemplazo; cv set con los puntos no escogidos)
# Más bias que el 10 fold cross-validation (uso menos datos para training)
# pero menos varianza (uso más datos para test)
set.seed(22)
training.splits <- createResample(y = all.training.data$Sales, times = 3)
training.set.1 <- all.training.data[training.splits[[1]], ]
cv.set.1 <- all.training.data[-training.splits[[1]], ]
training.set.2 <- all.training.data[training.splits[[2]], ]
cv.set.2 <- all.training.data[-training.splits[[2]], ]
training.set.3 <- all.training.data[training.splits[[3]], ]
cv.set.3 <- all.training.data[-training.splits[[3]], ]



# Si quiero crear cv sets con máxima disimilitud (cuando tengo muy pocos datos
# y quiero que el test sea representativo de todo el espacio) uso la función
# maxdissim. En el libro no viene ningún ejemplo.



################################################################################
# Ejemplo: cómo elegir los parámetros de un modelo utilizando estas técnicas
# El paquete "caret" contiene una función "train" que automatiza este proceso

set.seed(22)

# Voy a probar con k-nearest neighbour y otro de clasificación :P
training.rows <- createDataPartition(y = iris$Species, p = 0.85, list = FALSE)
training <- iris[training.rows, ]
test <- iris[-training.rows, ]

train.control <- trainControl(method = "repeatedcv", 
                              number = 10, repeats = 10)
k.grid <- data.frame("k" = 1:10)
knn.model <- train(Species ~ ., data = iris, method = "knn", 
                   preProc = c("center", "scale"), 
                   trControl = train.control,
                   tuneGrid = k.grid)
plot(knn.model)

l.grid <- data.frame("nIter" = seq.int(from = 1, to = 50, by = 1))
logistic.regression.model <- train(Species ~ ., data = iris, method = "LogitBoost", 
                                   preProc = c("center", "scale"), 
                                   trControl = train.control,
                                   tuneGrid = l.grid)
plot(logistic.regression.model)

prediccion.knn <- predict(knn.model, test %>% select(-Species))
prediccion.l <- predict(logistic.regression.model, test %>% select(-Species))
realidad <- test$Species
cat("accuracy.knn = ", sum(prediccion.knn == realidad)/length(realidad))
cat("accuracy.l = ", sum(prediccion.l == realidad)/length(realidad))



# Comparación de los dos modelos usando estadísticas de cv: ¿cuál es mejor?
resamp <- resamples(list(KNN = knn.model, LOGISTIC = logistic.regression.model))
summary(resamp)
diferencia.entre.modelos <- diff(resamp)
summary(diferencia.entre.modelos)
# Suponiendo que los modelos son iguales (hipótesis nula), el p-valor es la
# probabilidad de obtener los resultados que hemos obtenido. Si es alto significa
# que no podemos suponer que la hipótesis nula es falsa. Esto significa que los
# modelos no tienen un comportamiento significativamente distinto.
