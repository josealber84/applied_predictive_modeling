# Clean data and load packages
source("get_and_clean_data.R")
library(caret)
library(lubridate)
library(doParallel)

# Semilla!
set.seed(22)

# Cojo unos cuantos datos, que todos son muchos :)
pocos.datos.rows <- createDataPartition(all.training.data[, 1], p = 0.2, list = F)
pocos.datos <- all.training.data[pocos.datos.rows, ]
resto.datos <- all.training.data[-pocos.datos.rows, ]
test.rows <- createDataPartition(y = resto.datos[, 1], p = 0.01, list = FALSE)
test <- resto.datos[test.rows, ]
rm(all.training.data)


################################################################################

# Hago un modelo lineal sencillo
set.seed(22)

# Multicore configuration
n.cores <- detectCores()
cl <- makeCluster(n.cores)
registerDoParallel(cl)

# Parameter fitting
start <- now()
train.control <- trainControl(method = "cv", number = 10, verboseIter = TRUE)
model <- train(Sales ~ ., data = pocos.datos, method = "lm", 
               preProc = c("center", "scale"), 
               trControl = train.control,
               verbose = TRUE)
end <- now()
cat("\nHe tardado", end-start, "en hacer los calculos!\n")


# Stop multicore
stopCluster(cl)

# prediccion.knn.3 <- predict(knn.model, test [, -1])

################################################################################

# Hago un modelo KNN sencillo
set.seed(22)

# Multicore configuration
n.cores <- detectCores()
cl <- makeCluster(n.cores)
registerDoParallel(cl)

# Parameter fitting
start <- now()
train.control <- trainControl(method = "cv", number = 10, verboseIter = TRUE)
k.grid <- data.frame("k" = c(5))
knn.model <- train(x = pocos.datos[, -1], y = pocos.datos[, 1], method = "knn", 
                   preProc = c("center", "scale"), 
                   trControl = train.control,
                   tuneGrid = k.grid,
                   verbose = TRUE)
end <- now()
cat("\nHe tardado", end-start, "en hacer los calculos!\n")


# Stop multicore
stopCluster(cl)

# prediccion.knn.3 <- predict(knn.model, test [, -1])

################################################################################