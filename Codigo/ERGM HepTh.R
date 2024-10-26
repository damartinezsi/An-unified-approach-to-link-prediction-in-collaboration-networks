# Librerías --------------------------------------------------------------------
suppressMessages(suppressWarnings(library("igraph")))
suppressMessages(suppressWarnings(library("ergm")))
suppressMessages(suppressWarnings(library("network")))
suppressMessages(suppressWarnings(library("intergraph")))
suppressMessages(suppressWarnings(library("doParallel")))
suppressMessages(suppressWarnings(library("pROC")))
suppressMessages(suppressWarnings(library("Rglpk")))


# Definimos el directorio de trabajo
setwd("~/Git Repositories/SNA")


# Definimos una Semilla
set.seed(777) 


# Importamos la base de datos --------------------------------------------------

base <- read.delim("CA-HepTh.txt", comment.char="#")


# Renombramos las columnas
colnames(base) <- c("from", "to")


# Definimos la red como un objeto igraph
g <- igraph::graph_from_data_frame(d = base, directed = FALSE)


# Verificamos los atributos básicos de la red
igraph::ecount(g)
igraph::vcount(g)
igraph::is_directed(g)
igraph::is_simple(g)


# Simplificamos la red
g <- simplify(g, remove.multiple = TRUE, remove.loops = TRUE)


# Guardamos el grafo como un objeto del tipo network
net <- asNetwork(g)


# Liberamos memoria RAM
rm(g, base)
gc()


# Ajustamos el modelo ERGM -----------------------------------------------------


# Modelo 
start_time <- Sys.time()
modelo_simple <- ergm(net ~ edges)
end_time <- Sys.time()
print(end_time - start_time)


# Guardamos el modelo
save(modelo_simple, file = "ERGM_B5.RData")


# Simulamos redes a partir del modelo ajustado
start_time <- Sys.time()
num_sim <- 50
sim_nets <- simulate(modelo_simple, nsim = num_sim, output = "network", seed = 777)
end_time <- Sys.time()
print(end_time - start_time)


# Obtenemos la matriz de adyacencia de la red observada
observed <- as.matrix.network(net)


# Inicializamos una matriz para acumular las simulaciones
predicted_prob <- matrix(0, nrow = network.size(net), ncol = network.size(net))


# Acumulamos las matrices de adyacencia simuladas
start_time <- Sys.time()
conteo = 1
for (i in 1:num_sim) {
  print(conteo)
  sim_mat <- as.matrix.network(sim_nets[[i]])
  predicted_prob <- predicted_prob + sim_mat
  conteo = conteo + 1
}
end_time <- Sys.time()
print(end_time - start_time)


# Calculamos la probabilidad promedio de los enlaces
predicted_prob <- predicted_prob / num_sim


# Aplanamos las matrices para calcular el AUC
predicted_prob <- predicted_prob[upper.tri(predicted_prob)]
observed <- observed[upper.tri(observed)]


# Guardamos las probabilidades
write.csv(data.frame(valor = predicted_prob), file = "predict_HepTh.csv", row.names = FALSE)
write.csv(data.frame(valor = observed), file = "observed_HepTh.csv", row.names = FALSE)


rm(modelo_simple, net, sim_mat, sim_nets)
gc()


# Calculamos el AUC
roc_curve <- roc(observed, predicted_prob)
auc_value <- auc(roc_curve)


# Mostramos el valor del AUC
print(paste("AUC:", auc_value))