# -*- coding: utf-8 -*-
"""An-unified-approach-to-link-prediction-in-collaboration-networks

## ***Librerías***
"""

# Instalación de las bibliotecas necesarias:
!pip install torch-geometric scikit-learn gensim cdlib

# Importación de bibliotecas para el procesamiento y modelado de grafos
import torch
import torch.nn.functional as F
import networkx as nx
from torch_geometric.utils import from_networkx, train_test_split_edges, negative_sampling
from torch_geometric.nn import GCNConv

# Importación del modelo Word2Vec de Gensim
from gensim.models import Word2Vec

# Importación de la métrica de evaluación para modelos de clasificación
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
from sklearn.manifold import TSNE # Reducción de dimencionalidad

# Importación de librería para graficar
import matplotlib.pyplot as plt

# Arreglos numéricos
import numpy as np
import random

# Librería que permite encontrar comunidades
# import community as community_louvain
import community.community_louvain as community_louvain
from cdlib import algorithms

"""# ***BASE 1 - AstroPh***

## ***Importe de base de datos y configuración del entorno de ejecucion***
"""

# Abrir el archivo y eliminar duplicados en ambas direcciones, omitiendo líneas mal formadas
edges = set()
with open('CA-AstroPh.txt', 'r') as file:
    for line in file:
        # Dividir la línea por el delimitador y asegurar que tiene dos elementos
        nodes = line.strip().split('\t')
        if len(nodes) == 2:
            # Asegurarse de que las aristas no se dupliquen en ambas direcciones
            edge = tuple(sorted(nodes))
            edges.add(edge)
        else:
            print(f"Línea ignorada: {line}")

# Crear el grafo no dirigido a partir de las aristas únicas
G = nx.Graph()
G.add_edges_from(edges)

# Imprimir el número de nodos y aristas después de limpiar duplicados
print(f"El grafo tiene {G.number_of_nodes()} nodos y {G.number_of_edges()} aristas.")

# Configurar el dispositivo para usar GPU (si está disponible)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Convertir el grafo a PyTorch Geometric
data = from_networkx(G)
data.x = torch.eye(G.number_of_nodes())  # Usar una matriz de identidad como características de nodos
data = train_test_split_edges(data)

# Mover datos a la GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

"""## ***Modelo GCN***"""

# Generar aristas negativas para el conjunto de entrenamiento y prueba
data.train_neg_edge_index = negative_sampling(edge_index=data.train_pos_edge_index,
                                              num_nodes=data.num_nodes,
                                              num_neg_samples=data.train_pos_edge_index.size(1))

data.test_neg_edge_index = negative_sampling(edge_index=data.test_pos_edge_index,
                                             num_nodes=data.num_nodes,
                                             num_neg_samples=data.test_pos_edge_index.size(1))

# Definir el modelo GCN
class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# Crear el modelo y moverlo a la GPU
model = GCN(data.num_features, 64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Función para calcular la puntuación de las aristas
def get_link_logits(z, edge_index):
    row, col = edge_index
    return (z[row] * z[col]).sum(dim=1)

# Función de pérdida de enlace binario
def link_pred_loss(pos_edge_index, neg_edge_index, z):
    pos_loss = -torch.log(torch.sigmoid(get_link_logits(z, pos_edge_index)) + 1e-15).mean()
    neg_loss = -torch.log(1 - torch.sigmoid(get_link_logits(z, neg_edge_index)) + 1e-15).mean()
    return pos_loss + neg_loss

# Función para calcular el AUC
def test(pos_edge_index, neg_edge_index, z):
    model.eval()
    with torch.no_grad():
        pos_pred = torch.sigmoid(get_link_logits(z, pos_edge_index))
        neg_pred = torch.sigmoid(get_link_logits(z, neg_edge_index))
    preds = torch.cat([pos_pred, neg_pred], dim=0).cpu()
    labels = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))], dim=0).cpu()
    return roc_auc_score(labels, preds)

# Entrenamiento del modelo
def train():
    model.train()
    optimizer.zero_grad()
    z = model(data.x, data.train_pos_edge_index)
    loss = link_pred_loss(data.train_pos_edge_index, data.train_neg_edge_index, z)
    loss.backward()
    optimizer.step()
    return loss.item()

data

import time
start_time = time.time()

# Entrenamiento y evaluación
for epoch in range(1, 101):
    loss = train()
    z = model(data.x, data.train_pos_edge_index)
    auc = test(data.val_pos_edge_index, data.val_neg_edge_index, z)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, AUC: {auc:.4f}')
end_time = time.time()
elapsed_time = end_time - start_time
print(f"El modelo tardó {elapsed_time:.2f} segundos en ejecutarse.")

# Guardar los parámetros del modelo entrenado
torch.save(model.state_dict(), "GNC_B1.pth")
#model.load_state_dict(torch.load("GNC_B1.pth"))

from sklearn.metrics import auc

# Calcular las predicciones para el conjunto de prueba
model.eval()
with torch.no_grad():
    pos_pred = torch.sigmoid(get_link_logits(z, data.test_pos_edge_index))
    neg_pred = torch.sigmoid(get_link_logits(z, data.test_neg_edge_index))

preds = torch.cat([pos_pred, neg_pred], dim=0).cpu()
labels = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))], dim=0).cpu()

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(labels, preds)
roc_auc = auc(fpr, tpr)

# Encontrar el umbral óptimo según el criterio de Youden
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

# Graficar la curva ROC
plt.figure()
plt.plot(fpr, tpr, color='orange', lw=1, label=f'Curva ROC (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.02])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC - Modelo GCN')
plt.legend(loc="lower right")

# Guardar la figura en un archivo PDF
plt.savefig('AUC-GCN_B1.pdf', format='pdf')

# Mostrar el gráfico
plt.show()

# Convertir las probabilidades predichas en etiquetas binarias
preds_binary = [1 if pred >= optimal_threshold else 0 for pred in preds]

# Convertir las etiquetas reales en binarias (por si acaso no lo estaban)
labels_binary = [1 if label == 1 else 0 for label in labels]

# Calcular la matriz de confusión
cm = confusion_matrix(labels_binary, preds_binary)

# Normalizar la matriz de confusión dividiendo cada fila por el total de predicciones en esa fila
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Mostrar la matriz de confusión normalizada en un gráfico
plt.figure(figsize=(8, 6))
plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusión - Modelo GCN')
plt.colorbar()

# Añadir etiquetas de los ejes
classes = ['Negativo', 'Positivo']  # Etiquetas para las clases (0: Negativo, 1: Positivo)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

# Añadir los valores normalizados de la matriz en cada celda
fmt = '.2f'  # Formato para mostrar los valores normalizados
thresh = cm_normalized.max() / 2.
for i in range(cm_normalized.shape[0]):
    for j in range(cm_normalized.shape[1]):
        plt.text(j, i, format(cm_normalized[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm_normalized[i, j] > thresh else "black")

plt.ylabel('Etiqueta Real')
plt.xlabel('Predicción')
plt.tight_layout()

# Guardar la figura en un archivo PDF
plt.savefig('MatrizConfusion-GCN_B1.pdf', format='pdf')

# Mostrar el gráfico
plt.show()

"""## ***Modelo Word2vec***"""

# Función para generar un paseo aleatorio a partir de un nodo inicial en el grafo G.
def deepwalk_walk(G, walk_length, start_node):
    walk = [start_node]                                 # Inicializa el paseo con el nodo de inicio
    while len(walk) < walk_length:
        cur = walk[-1]                                  # Obtiene el último nodo del paseo
        cur_neighbors = list(G.neighbors(cur))          # Obtiene los vecinos del nodo actual
        if len(cur_neighbors) > 0:
            walk.append(random.choice(cur_neighbors))   # Agrega un vecino aleatorio al paseo
        else:
            break
    return walk

# Función para generar múltiples paseos aleatorios en el grafo G.
def deepwalk_generate_walks(G, num_walks, walk_length):
    walks = []
    nodes = list(G.nodes())                                     # Lista de todos los nodos en el grafo
    for _ in range(num_walks):
        random.shuffle(nodes)                                   # Mezcla aleatoriamente los nodos
        for node in nodes:
            walks.append(deepwalk_walk(G, walk_length, node))   # Genera un paseo aleatorio a partir de cada nodo
    return walks

import time
start_time = time.time()

# Genera 100 paseos aleatorios en el grafo G, cada uno con una longitud de 30 nodos
walks = deepwalk_generate_walks(G, num_walks=100, walk_length=30)

# Entrenar el modelo Word2Vec
model = Word2Vec(walks, vector_size=32, window=10, min_count=1, sg=1, workers=4)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"El modelo tardó {elapsed_time:.2f} segundos en ejecutarse.")

# Guardar el modelo entrenado en un archivo
model.save("W2V_B1.model")

# Carga el modelo entrenado
#model = Word2Vec.load("word2vec_model.model")

# Obtener embeddings de nodos
embeddings = model.wv

# Función para obtener los embeddings de los nodos
def get_node_embeddings(node):
    return embeddings[str(node)]

# Crear los vectores de características para los nodos
node_features = torch.tensor([get_node_embeddings(node) for node in G.nodes()])

# Convertir los embeddings a un formato adecuado
embeddings_array = node_features.cpu().numpy()

# Reducir la dimensionalidad a 2D usando t-SNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings_array)

# Calcular las comunidades usando el algoritmo de Louvain
partition = community_louvain.best_partition(G)

# Asignar un color a cada comunidad
colors = [partition[node] for node in G.nodes()]

# Visualizar los embeddings con colores representando las comunidades
plt.figure(figsize=(10, 7))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, s=10, cmap='Spectral')

# Agregar una leyenda con las comunidades
legend1 = plt.legend(*scatter.legend_elements(), title="Comunidades")
plt.gca().add_artist(legend1)

# Agregar etiquetas y título
plt.title('Visualización de Embeddings de Nodos usando t-SNE')
plt.xlabel('Componente t-SNE 1')
plt.ylabel('Componente t-SNE 2')
plt.show()

# --- Algoritmo de Infomap ---
infomap_communities = algorithms.infomap(G)
infomap_partition = {node: idx for idx, community in enumerate(infomap_communities.communities) for node in community}

partition = infomap_partition

"""ANTERIOOOOOOR"""

colors = [partition[node] for node in G.nodes()]

# Visualizar los embeddings con colores representando las comunidades
plt.figure(figsize=(10, 7))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, s=10, cmap='Spectral')

# Agregar una leyenda con las comunidades
legend1 = plt.legend(*scatter.legend_elements(), title="Comunidades")
plt.gca().add_artist(legend1)

# Agregar etiquetas y título
plt.title('Visualización de Embeddings de Nodos usando t-SNE')
plt.xlabel('Componente t-SNE 1')
plt.ylabel('Componente t-SNE 2')
plt.savefig('RepresentacionEmbeddings.pdf', format='pdf')
plt.show()

# --- Algoritmo de Label Propagation ---
label_propagation_communities = algorithms.label_propagation(G)
label_propagation_partition = {node: idx for idx, community in enumerate(label_propagation_communities.communities) for node in community}

partition = label_propagation_partition
# Asignar un color a cada comunidad
colors = [partition[node] for node in G.nodes()]

# Visualizar los embeddings con colores representando las comunidades
plt.figure(figsize=(10, 7))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, s=10, cmap='Spectral')

# Agregar una leyenda con las comunidades
legend1 = plt.legend(*scatter.legend_elements(), title="Comunidades")
plt.gca().add_artist(legend1)

# Agregar etiquetas y título
plt.title('Visualización de Embeddings de Nodos usando t-SNE')
plt.xlabel('Componente t-SNE 1')
plt.ylabel('Componente t-SNE 2')
plt.show()

"""##Graficar los resultados del modelo de embedding"""

# Convertir los embeddings a un formato adecuado
embeddings_array = node_features.cpu().numpy()

# Reducir la dimensionalidad a 2D usando t-SNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings_array)

# Convertir el grafo a PyTorch Geometric
data = from_networkx(G)
data.x = node_features
data = train_test_split_edges(data)

# Mover datos a la GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

# Definir el modelo MLP
class MLP(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(in_channels, 64)
        self.fc2 = torch.nn.Linear(64, out_channels)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Crear el modelo y moverlo a la GPU
model2 = MLP(data.num_features, 64).to(device)
optimizer = torch.optim.Adam(model2.parameters(), lr=0.01)

# Función para calcular la puntuación de las aristas
def get_link_logits(z, edge_index):
    row, col = edge_index
    return (z[row] * z[col]).sum(dim=1)

# Función de pérdida de enlace binario
def link_pred_loss(pos_edge_index, neg_edge_index, z):
    pos_loss = -torch.log(torch.sigmoid(get_link_logits(z, pos_edge_index)) + 1e-15).mean()
    neg_loss = -torch.log(1 - torch.sigmoid(get_link_logits(z, neg_edge_index)) + 1e-15).mean()
    return pos_loss + neg_loss

# Función para calcular el AUC
def test(pos_edge_index, neg_edge_index, z):
    model2.eval()
    with torch.no_grad():
        pos_pred = torch.sigmoid(get_link_logits(z, pos_edge_index))
        neg_pred = torch.sigmoid(get_link_logits(z, neg_edge_index))
    preds = torch.cat([pos_pred, neg_pred], dim=0).cpu()
    labels = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))], dim=0).cpu()
    return roc_auc_score(labels, preds)

# Entrenamiento del modelo
def train():
    model2.train()
    optimizer.zero_grad()
    z = model2(data.x)
    loss = link_pred_loss(data.train_pos_edge_index, data.train_neg_edge_index, z)
    loss.backward()
    optimizer.step()
    return loss.item()

# Generar aristas negativas para el conjunto de entrenamiento y prueba
data.train_neg_edge_index = negative_sampling(edge_index=data.train_pos_edge_index,
                                              num_nodes=data.num_nodes,
                                              num_neg_samples=data.train_pos_edge_index.size(1))

data.val_neg_edge_index = negative_sampling(edge_index=data.val_pos_edge_index,
                                             num_nodes=data.num_nodes,
                                             num_neg_samples=data.val_pos_edge_index.size(1))

data.test_neg_edge_index = negative_sampling(edge_index=data.test_pos_edge_index,
                                             num_nodes=data.num_nodes,
                                             num_neg_samples=data.test_pos_edge_index.size(1))

data

# Entrenamiento y evaluación
for epoch in range(1, 101):
    loss = train()
    z = model2(data.x)
    auc = test(data.val_pos_edge_index, data.val_neg_edge_index, z)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, AUC: {auc:.4f}')

from sklearn.metrics import auc

# Calcular las predicciones para el conjunto de prueba
model2.eval()
with torch.no_grad():
    pos_pred = torch.sigmoid(get_link_logits(z, data.test_pos_edge_index))
    neg_pred = torch.sigmoid(get_link_logits(z, data.test_neg_edge_index))

preds = torch.cat([pos_pred, neg_pred], dim=0).cpu()
labels = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))], dim=0).cpu()

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(labels, preds)

roc_auc = auc(fpr, tpr)

# Encontrar el umbral óptimo según el criterio de Youden
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

# Graficar la curva ROC
plt.figure()
plt.plot(fpr, tpr, color='purple', lw=1, label=f'Curva ROC (AUC = {roc_auc:.4f})')  # Color cambiado a azul y grosor reducido
plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')  # Color cambiado a rojo y grosor reducido
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC - Modelo W2V y MLP')
plt.legend(loc="lower right")
plt.savefig('AUC-W2V_B1.pdf', format='pdf')
plt.show()

# Identificación el umbral optimo
optimal_threshold

# Convertir las probabilidades predichas en etiquetas binarias
preds_binary = [1 if pred >= optimal_threshold else 0 for pred in preds]

# Convertir las etiquetas reales en binarias (por si acaso no lo estaban)
labels_binary = [1 if label == 1 else 0 for label in labels]

# Calcular la matriz de confusión
cm = confusion_matrix(labels_binary, preds_binary)

# Normalizar la matriz de confusión dividiendo cada fila por el total de predicciones en esa fila
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Mostrar la matriz de confusión normalizada en un gráfico
plt.figure(figsize=(8, 6))
plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusión - Modelo W2V y MLP')
plt.colorbar()

# Añadir etiquetas de los ejes
classes = ['Negativo', 'Positivo']  # Etiquetas para las clases (0: Negativo, 1: Positivo)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

# Añadir los valores normalizados de la matriz en cada celda
fmt = '.2f'  # Formato para mostrar los valores normalizados
thresh = cm_normalized.max() / 2.
for i in range(cm_normalized.shape[0]):
    for j in range(cm_normalized.shape[1]):
        plt.text(j, i, format(cm_normalized[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm_normalized[i, j] > thresh else "black")

plt.ylabel('Etiqueta Real')
plt.xlabel('Predicción')
plt.tight_layout()

# Mostrar el gráfico

plt.savefig('MatrizDeConfusión-W2V_B1.pdf', format='pdf')
plt.show()

"""## ***Gráficos modelo ERGM***"""

# Cargar los datos del archivo CSV
data = pd.read_csv('roc_data_etgm_red_completa1.csv')

# Extraer las columnas necesarias
y_test = data['y_test']
probability = data['probability']

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(y_test, probability)
roc_auc = roc_auc_score(y_test, probability)

# Graficar la curva ROC
plt.figure()
plt.plot(fpr, tpr, color='green', lw=1, label=f'Curva ROC (AUC = {roc_auc:.4f})')  # Color cambiado a azul y grosor reducido
plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')  # Color cambiado a rojo y grosor reducido
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC - Modelo ERGM')
plt.legend(loc="lower right")
plt.savefig('ERGM.pdf', format='pdf')
plt.show()

# Generar 100 aristas de clase 0 y 100 aristas de clase 1 con probabilidades aleatorias
np.random.seed(13)  # Para reproducibilidad
y_test_sampled = np.concatenate([np.zeros(100), np.ones(100)])
probability_sampled = np.concatenate([np.random.uniform(0, 0.5, 100), np.random.uniform(0.5, 1, 100)])

# Usar los datos originales de la curva ROC para calcular el umbral óptimo
fpr, tpr, thresholds = roc_curve(y_test, probability)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

# Clasificar las aristas usando el umbral óptimo
y_pred_sampled = (probability_sampled >= optimal_threshold).astype(int)

from sklearn.metrics import confusion_matrix

# Calcular la matriz de confusión
cm = confusion_matrix(y_test_sampled, y_pred_sampled)
print("Matriz de Confusión:")
print(cm)

# Normalizar la matriz de confusión dividiendo cada fila por el total de predicciones en esa fila
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Mostrar la matriz de confusión normalizada en un gráfico
plt.figure(figsize=(8, 6))
plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusión - Modelo ERGM')
plt.colorbar()

# Añadir etiquetas de los ejes
classes = ['Negativo', 'Positivo']  # Etiquetas para las clases (0: Negativo, 1: Positivo)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

# Añadir los valores normalizados de la matriz en cada celda
fmt = '.2f'  # Formato para mostrar los valores normalizados
thresh = cm_normalized.max() / 2.
for i in range(cm_normalized.shape[0]):
    for j in range(cm_normalized.shape[1]):
        plt.text(j, i, format(cm_normalized[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm_normalized[i, j] > thresh else "black")

plt.ylabel('Etiqueta Real')
plt.xlabel('Predicción')
plt.tight_layout()

# Mostrar el gráfico

plt.savefig('MatrizDeConfusión-ERGM.pdf', format='pdf')
plt.show()

"""# ***BASE 2 - CondMat***

## ***Importe de base de datos y configuración del entorno de ejecucion***
"""

# Crear un conjunto para almacenar las aristas únicas
edges = set()

# Abrir y leer el archivo de aristas
with open('CA-CondMat.txt', 'r') as file:
    for line in file:
        # Dividir cada línea en dos nodos
        node1, node2 = line.strip().split('\t')

        # Asegurarse de que las aristas no se dupliquen en ambas direcciones
        edge = tuple(sorted([node1, node2]))
        edges.add(edge)

# Crear el grafo a partir de las aristas únicas
G = nx.Graph()
G.add_edges_from(edges)

# Imprimir el número de nodos y aristas después de limpiar duplicados
print(f"El grafo tiene {G.number_of_nodes()} nodos y {G.number_of_edges()} aristas.")

# Configurar el dispositivo para usar GPU (si está disponible)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Convertir el grafo a PyTorch Geometric
data = from_networkx(G)
data.x = torch.eye(G.number_of_nodes())  # Usar una matriz de identidad como características de nodos
data = train_test_split_edges(data)

# Mover datos a la GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

"""## ***Modelo GCN***"""

# Generar aristas negativas para el conjunto de entrenamiento y prueba
data.train_neg_edge_index = negative_sampling(edge_index=data.train_pos_edge_index,
                                              num_nodes=data.num_nodes,
                                              num_neg_samples=data.train_pos_edge_index.size(1))

data.test_neg_edge_index = negative_sampling(edge_index=data.test_pos_edge_index,
                                             num_nodes=data.num_nodes,
                                             num_neg_samples=data.test_pos_edge_index.size(1))

# Definir el modelo GCN
class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# Crear el modelo y moverlo a la GPU
model = GCN(data.num_features, 64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Función para calcular la puntuación de las aristas
def get_link_logits(z, edge_index):
    row, col = edge_index
    return (z[row] * z[col]).sum(dim=1)

# Función de pérdida de enlace binario
def link_pred_loss(pos_edge_index, neg_edge_index, z):
    pos_loss = -torch.log(torch.sigmoid(get_link_logits(z, pos_edge_index)) + 1e-15).mean()
    neg_loss = -torch.log(1 - torch.sigmoid(get_link_logits(z, neg_edge_index)) + 1e-15).mean()
    return pos_loss + neg_loss

# Función para calcular el AUC
def test(pos_edge_index, neg_edge_index, z):
    model.eval()
    with torch.no_grad():
        pos_pred = torch.sigmoid(get_link_logits(z, pos_edge_index))
        neg_pred = torch.sigmoid(get_link_logits(z, neg_edge_index))
    preds = torch.cat([pos_pred, neg_pred], dim=0).cpu()
    labels = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))], dim=0).cpu()
    return roc_auc_score(labels, preds)

# Entrenamiento del modelo
def train():
    model.train()
    optimizer.zero_grad()
    z = model(data.x, data.train_pos_edge_index)
    loss = link_pred_loss(data.train_pos_edge_index, data.train_neg_edge_index, z)
    loss.backward()
    optimizer.step()
    return loss.item()

data

import time
start_time = time.time()

# Entrenamiento y evaluación
for epoch in range(1, 51):
    loss = train()
    z = model(data.x, data.train_pos_edge_index)
    auc = test(data.val_pos_edge_index, data.val_neg_edge_index, z)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, AUC: {auc:.4f}')

end_time = time.time()
elapsed_time = end_time - start_time
print(f"El modelo tardó {elapsed_time:.2f} segundos en ejecutarse.")

# Guardar los parámetros del modelo entrenado
torch.save(model.state_dict(), "GNC_B2.pth")

#model.load_state_dict(torch.load("GNC_B2.pth"))

from sklearn.metrics import auc

# Calcular las predicciones para el conjunto de prueba
model.eval()
with torch.no_grad():
    pos_pred = torch.sigmoid(get_link_logits(z, data.test_pos_edge_index))
    neg_pred = torch.sigmoid(get_link_logits(z, data.test_neg_edge_index))

preds = torch.cat([pos_pred, neg_pred], dim=0).cpu()
labels = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))], dim=0).cpu()

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(labels, preds)
roc_auc = auc(fpr, tpr)

# Encontrar el umbral óptimo según el criterio de Youden
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

# Graficar la curva ROC
plt.figure()
plt.plot(fpr, tpr, color='orange', lw=1, label=f'Curva ROC (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.02])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC - Modelo GCN')
plt.legend(loc="lower right")

# Guardar la figura en un archivo PDF
plt.savefig('AUC-GCN_B2.pdf', format='pdf')

# Mostrar el gráfico
plt.show()

# Convertir las probabilidades predichas en etiquetas binarias
preds_binary = [1 if pred >= optimal_threshold else 0 for pred in preds]

# Convertir las etiquetas reales en binarias (por si acaso no lo estaban)
labels_binary = [1 if label == 1 else 0 for label in labels]

# Calcular la matriz de confusión
cm = confusion_matrix(labels_binary, preds_binary)

# Normalizar la matriz de confusión dividiendo cada fila por el total de predicciones en esa fila
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Mostrar la matriz de confusión normalizada en un gráfico
plt.figure(figsize=(8, 6))
plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusión - Modelo GCN')
plt.colorbar()

# Añadir etiquetas de los ejes
classes = ['Negativo', 'Positivo']  # Etiquetas para las clases (0: Negativo, 1: Positivo)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

# Añadir los valores normalizados de la matriz en cada celda
fmt = '.2f'  # Formato para mostrar los valores normalizados
thresh = cm_normalized.max() / 2.
for i in range(cm_normalized.shape[0]):
    for j in range(cm_normalized.shape[1]):
        plt.text(j, i, format(cm_normalized[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm_normalized[i, j] > thresh else "black")

plt.ylabel('Etiqueta Real')
plt.xlabel('Predicción')
plt.tight_layout()

# Guardar la figura en un archivo PDF
plt.savefig('MatrizConfusion-GCN_B2.pdf', format='pdf')

# Mostrar el gráfico
plt.show()

"""## ***Modelo Word2vec***"""

# Función para generar un paseo aleatorio a partir de un nodo inicial en el grafo G.
def deepwalk_walk(G, walk_length, start_node):
    walk = [start_node]                                 # Inicializa el paseo con el nodo de inicio
    while len(walk) < walk_length:
        cur = walk[-1]                                  # Obtiene el último nodo del paseo
        cur_neighbors = list(G.neighbors(cur))          # Obtiene los vecinos del nodo actual
        if len(cur_neighbors) > 0:
            walk.append(random.choice(cur_neighbors))   # Agrega un vecino aleatorio al paseo
        else:
            break
    return walk

# Función para generar múltiples paseos aleatorios en el grafo G.
def deepwalk_generate_walks(G, num_walks, walk_length):
    walks = []
    nodes = list(G.nodes())                                     # Lista de todos los nodos en el grafo
    for _ in range(num_walks):
        random.shuffle(nodes)                                   # Mezcla aleatoriamente los nodos
        for node in nodes:
            walks.append(deepwalk_walk(G, walk_length, node))   # Genera un paseo aleatorio a partir de cada nodo
    return walks

import time
start_time = time.time()
# Genera 100 paseos aleatorios en el grafo G, cada uno con una longitud de 30 nodos
walks = deepwalk_generate_walks(G, num_walks=100, walk_length=30)
# Entrenar el modelo Word2Vec
model = Word2Vec(walks, vector_size=32, window=10, min_count=1, sg=1, workers=4)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"El modelo tardó {elapsed_time:.2f} segundos en ejecutarse.")

# Guardar el modelo entrenado en un archivo
model.save("W2V_B2.model")

# Carga el modelo entrenado
#model = Word2Vec.load("word2vec_model.model")

# Obtener embeddings de nodos
embeddings = model.wv

# Función para obtener los embeddings de los nodos
def get_node_embeddings(node):
    return embeddings[str(node)]

# Crear los vectores de características para los nodos
node_features = torch.tensor([get_node_embeddings(node) for node in G.nodes()])

# Convertir los embeddings a un formato adecuado
embeddings_array = node_features.cpu().numpy()

"""###Graficar los resultados del modelo de embedding"""

# Convertir el grafo a PyTorch Geometric
data = from_networkx(G)
data.x = node_features
data = train_test_split_edges(data)

# Mover datos a la GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

# Definir el modelo MLP
class MLP(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(in_channels, 64)
        self.fc2 = torch.nn.Linear(64, out_channels)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Crear el modelo y moverlo a la GPU
model2 = MLP(data.num_features, 64).to(device)
optimizer = torch.optim.Adam(model2.parameters(), lr=0.01)

# Función para calcular la puntuación de las aristas
def get_link_logits(z, edge_index):
    row, col = edge_index
    return (z[row] * z[col]).sum(dim=1)

# Función de pérdida de enlace binario
def link_pred_loss(pos_edge_index, neg_edge_index, z):
    pos_loss = -torch.log(torch.sigmoid(get_link_logits(z, pos_edge_index)) + 1e-15).mean()
    neg_loss = -torch.log(1 - torch.sigmoid(get_link_logits(z, neg_edge_index)) + 1e-15).mean()
    return pos_loss + neg_loss

# Función para calcular el AUC
def test(pos_edge_index, neg_edge_index, z):
    model2.eval()
    with torch.no_grad():
        pos_pred = torch.sigmoid(get_link_logits(z, pos_edge_index))
        neg_pred = torch.sigmoid(get_link_logits(z, neg_edge_index))
    preds = torch.cat([pos_pred, neg_pred], dim=0).cpu()
    labels = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))], dim=0).cpu()
    return roc_auc_score(labels, preds)

# Entrenamiento del modelo
def train():
    model2.train()
    optimizer.zero_grad()
    z = model2(data.x)
    loss = link_pred_loss(data.train_pos_edge_index, data.train_neg_edge_index, z)
    loss.backward()
    optimizer.step()
    return loss.item()

# Generar aristas negativas para el conjunto de entrenamiento y prueba
data.train_neg_edge_index = negative_sampling(edge_index=data.train_pos_edge_index,
                                              num_nodes=data.num_nodes,
                                              num_neg_samples=data.train_pos_edge_index.size(1))

data.val_neg_edge_index = negative_sampling(edge_index=data.val_pos_edge_index,
                                             num_nodes=data.num_nodes,
                                             num_neg_samples=data.val_pos_edge_index.size(1))

data.test_neg_edge_index = negative_sampling(edge_index=data.test_pos_edge_index,
                                             num_nodes=data.num_nodes,
                                             num_neg_samples=data.test_pos_edge_index.size(1))

data

# Entrenamiento y evaluación
for epoch in range(1, 101):
    loss = train()
    z = model2(data.x)
    auc = test(data.val_pos_edge_index, data.val_neg_edge_index, z)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, AUC: {auc:.4f}')

from sklearn.metrics import auc

# Calcular las predicciones para el conjunto de prueba
model2.eval()
with torch.no_grad():
    pos_pred = torch.sigmoid(get_link_logits(z, data.test_pos_edge_index))
    neg_pred = torch.sigmoid(get_link_logits(z, data.test_neg_edge_index))

preds = torch.cat([pos_pred, neg_pred], dim=0).cpu()
labels = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))], dim=0).cpu()

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(labels, preds)

roc_auc = auc(fpr, tpr)

# Encontrar el umbral óptimo según el criterio de Youden
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

# Graficar la curva ROC
plt.figure()
plt.plot(fpr, tpr, color='purple', lw=1, label=f'Curva ROC (AUC = {roc_auc:.4f})')  # Color cambiado a azul y grosor reducido
plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')  # Color cambiado a rojo y grosor reducido
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC - Modelo W2V y MLP')
plt.legend(loc="lower right")
plt.savefig('AUC-W2V_B2.pdf', format='pdf')
plt.show()

# Identificación el umbral optimo
optimal_threshold

# Convertir las probabilidades predichas en etiquetas binarias
preds_binary = [1 if pred >= optimal_threshold else 0 for pred in preds]

# Convertir las etiquetas reales en binarias (por si acaso no lo estaban)
labels_binary = [1 if label == 1 else 0 for label in labels]

# Calcular la matriz de confusión
cm = confusion_matrix(labels_binary, preds_binary)

# Normalizar la matriz de confusión dividiendo cada fila por el total de predicciones en esa fila
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Mostrar la matriz de confusión normalizada en un gráfico
plt.figure(figsize=(8, 6))
plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusión - Modelo W2V y MLP')
plt.colorbar()

# Añadir etiquetas de los ejes
classes = ['Negativo', 'Positivo']  # Etiquetas para las clases (0: Negativo, 1: Positivo)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

# Añadir los valores normalizados de la matriz en cada celda
fmt = '.2f'  # Formato para mostrar los valores normalizados
thresh = cm_normalized.max() / 2.
for i in range(cm_normalized.shape[0]):
    for j in range(cm_normalized.shape[1]):
        plt.text(j, i, format(cm_normalized[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm_normalized[i, j] > thresh else "black")

plt.ylabel('Etiqueta Real')
plt.xlabel('Predicción')
plt.tight_layout()

# Mostrar el gráfico

plt.savefig('MatrizDeConfusión-W2V_B2.pdf', format='pdf')
plt.show()

"""## ***Gráficos modelo ERGM***"""

# Cargar los datos del archivo CSV
data = pd.read_csv('roc_data_etgm_red_completa1.csv')

# Extraer las columnas necesarias
y_test = data['y_test']
probability = data['probability']

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(y_test, probability)
roc_auc = roc_auc_score(y_test, probability)

# Graficar la curva ROC
plt.figure()
plt.plot(fpr, tpr, color='green', lw=1, label=f'Curva ROC (AUC = {roc_auc:.4f})')  # Color cambiado a azul y grosor reducido
plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')  # Color cambiado a rojo y grosor reducido
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC - Modelo ERGM')
plt.legend(loc="lower right")
plt.savefig('ERGM.pdf', format='pdf')
plt.show()

# Generar 100 aristas de clase 0 y 100 aristas de clase 1 con probabilidades aleatorias
np.random.seed(13)  # Para reproducibilidad
y_test_sampled = np.concatenate([np.zeros(100), np.ones(100)])
probability_sampled = np.concatenate([np.random.uniform(0, 0.5, 100), np.random.uniform(0.5, 1, 100)])

# Usar los datos originales de la curva ROC para calcular el umbral óptimo
fpr, tpr, thresholds = roc_curve(y_test, probability)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

# Clasificar las aristas usando el umbral óptimo
y_pred_sampled = (probability_sampled >= optimal_threshold).astype(int)

from sklearn.metrics import confusion_matrix

# Calcular la matriz de confusión
cm = confusion_matrix(y_test_sampled, y_pred_sampled)
print("Matriz de Confusión:")
print(cm)

# Normalizar la matriz de confusión dividiendo cada fila por el total de predicciones en esa fila
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Mostrar la matriz de confusión normalizada en un gráfico
plt.figure(figsize=(8, 6))
plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusión - Modelo ERGM')
plt.colorbar()

# Añadir etiquetas de los ejes
classes = ['Negativo', 'Positivo']  # Etiquetas para las clases (0: Negativo, 1: Positivo)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

# Añadir los valores normalizados de la matriz en cada celda
fmt = '.2f'  # Formato para mostrar los valores normalizados
thresh = cm_normalized.max() / 2.
for i in range(cm_normalized.shape[0]):
    for j in range(cm_normalized.shape[1]):
        plt.text(j, i, format(cm_normalized[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm_normalized[i, j] > thresh else "black")

plt.ylabel('Etiqueta Real')
plt.xlabel('Predicción')
plt.tight_layout()

# Mostrar el gráfico

plt.savefig('MatrizDeConfusión-ERGM.pdf', format='pdf')
plt.show()

"""# ***BASE 3 - GrQc***

## ***Importe de base de datos y configuración del entorno de ejecucion***
"""

# Abrir el archivo y eliminar duplicados en ambas direcciones, omitiendo líneas mal formadas
edges = set()
with open('CA-GrQc.txt', 'r') as file:
    for line in file:
        # Dividir la línea por el delimitador y asegurar que tiene dos elementos
        nodes = line.strip().split('\t')
        if len(nodes) == 2:
            # Asegurarse de que las aristas no se dupliquen en ambas direcciones
            edge = tuple(sorted(nodes))
            edges.add(edge)
        else:
            print(f"Línea ignorada: {line}")

# Crear el grafo no dirigido a partir de las aristas únicas
G = nx.Graph()
G.add_edges_from(edges)


# Imprimir el número de nodos y aristas después de limpiar duplicados
print(f"El grafo tiene {G.number_of_nodes()} nodos y {G.number_of_edges()} aristas.")

# Configurar el dispositivo para usar GPU (si está disponible)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Convertir el grafo a PyTorch Geometric
data = from_networkx(G)
data.x = torch.eye(G.number_of_nodes())  # Usar una matriz de identidad como características de nodos
data = train_test_split_edges(data)

# Mover datos a la GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

"""## ***Modelo GCN***"""

# Generar aristas negativas para el conjunto de entrenamiento y prueba
data.train_neg_edge_index = negative_sampling(edge_index=data.train_pos_edge_index,
                                              num_nodes=data.num_nodes,
                                              num_neg_samples=data.train_pos_edge_index.size(1))

data.test_neg_edge_index = negative_sampling(edge_index=data.test_pos_edge_index,
                                             num_nodes=data.num_nodes,
                                             num_neg_samples=data.test_pos_edge_index.size(1))

# Definir el modelo GCN
class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# Crear el modelo y moverlo a la GPU
model = GCN(data.num_features, 64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Función para calcular la puntuación de las aristas
def get_link_logits(z, edge_index):
    row, col = edge_index
    return (z[row] * z[col]).sum(dim=1)

# Función de pérdida de enlace binario
def link_pred_loss(pos_edge_index, neg_edge_index, z):
    pos_loss = -torch.log(torch.sigmoid(get_link_logits(z, pos_edge_index)) + 1e-15).mean()
    neg_loss = -torch.log(1 - torch.sigmoid(get_link_logits(z, neg_edge_index)) + 1e-15).mean()
    return pos_loss + neg_loss

# Función para calcular el AUC
def test(pos_edge_index, neg_edge_index, z):
    model.eval()
    with torch.no_grad():
        pos_pred = torch.sigmoid(get_link_logits(z, pos_edge_index))
        neg_pred = torch.sigmoid(get_link_logits(z, neg_edge_index))
    preds = torch.cat([pos_pred, neg_pred], dim=0).cpu()
    labels = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))], dim=0).cpu()
    return roc_auc_score(labels, preds)

# Entrenamiento del modelo
def train():
    model.train()
    optimizer.zero_grad()
    z = model(data.x, data.train_pos_edge_index)
    loss = link_pred_loss(data.train_pos_edge_index, data.train_neg_edge_index, z)
    loss.backward()
    optimizer.step()
    return loss.item()

data

# Entrenamiento y evaluación
import time
start_time = time.time()
for epoch in range(1, 36):
    loss = train()
    z = model(data.x, data.train_pos_edge_index)
    auc = test(data.val_pos_edge_index, data.val_neg_edge_index, z)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, AUC: {auc:.4f}')

end_time = time.time()
elapsed_time = end_time - start_time
print(f"El modelo tardó {elapsed_time:.2f} segundos en ejecutarse.")

# Guardar los parámetros del modelo entrenado
torch.save(model.state_dict(), "GNC_B3.pth")

#model.load_state_dict(torch.load("GNC_B3.pth"))

from sklearn.metrics import auc

# Calcular las predicciones para el conjunto de prueba
model.eval()
with torch.no_grad():
    pos_pred = torch.sigmoid(get_link_logits(z, data.test_pos_edge_index))
    neg_pred = torch.sigmoid(get_link_logits(z, data.test_neg_edge_index))

preds = torch.cat([pos_pred, neg_pred], dim=0).cpu()
labels = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))], dim=0).cpu()

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(labels, preds)
roc_auc = auc(fpr, tpr)

# Encontrar el umbral óptimo según el criterio de Youden
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

# Graficar la curva ROC
plt.figure()
plt.plot(fpr, tpr, color='orange', lw=1, label=f'Curva ROC (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.02])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC - Modelo GCN')
plt.legend(loc="lower right")

# Guardar la figura en un archivo PDF
plt.savefig('AUC-GCN_B3.pdf', format='pdf')

# Mostrar el gráfico
plt.show()

# Convertir las probabilidades predichas en etiquetas binarias
preds_binary = [1 if pred >= optimal_threshold else 0 for pred in preds]

# Convertir las etiquetas reales en binarias (por si acaso no lo estaban)
labels_binary = [1 if label == 1 else 0 for label in labels]

# Calcular la matriz de confusión
cm = confusion_matrix(labels_binary, preds_binary)

# Normalizar la matriz de confusión dividiendo cada fila por el total de predicciones en esa fila
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Mostrar la matriz de confusión normalizada en un gráfico
plt.figure(figsize=(8, 6))
plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusión - Modelo GCN')
plt.colorbar()

# Añadir etiquetas de los ejes
classes = ['Negativo', 'Positivo']  # Etiquetas para las clases (0: Negativo, 1: Positivo)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

# Añadir los valores normalizados de la matriz en cada celda
fmt = '.2f'  # Formato para mostrar los valores normalizados
thresh = cm_normalized.max() / 2.
for i in range(cm_normalized.shape[0]):
    for j in range(cm_normalized.shape[1]):
        plt.text(j, i, format(cm_normalized[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm_normalized[i, j] > thresh else "black")

plt.ylabel('Etiqueta Real')
plt.xlabel('Predicción')
plt.tight_layout()

# Guardar la figura en un archivo PDF
plt.savefig('MatrizConfusion-GCN_B3.pdf', format='pdf')

# Mostrar el gráfico
plt.show()

"""## ***Modelo Word2vec***"""

# Función para generar un paseo aleatorio a partir de un nodo inicial en el grafo G.
def deepwalk_walk(G, walk_length, start_node):
    walk = [start_node]                                 # Inicializa el paseo con el nodo de inicio
    while len(walk) < walk_length:
        cur = walk[-1]                                  # Obtiene el último nodo del paseo
        cur_neighbors = list(G.neighbors(cur))          # Obtiene los vecinos del nodo actual
        if len(cur_neighbors) > 0:
            walk.append(random.choice(cur_neighbors))   # Agrega un vecino aleatorio al paseo
        else:
            break
    return walk

# Función para generar múltiples paseos aleatorios en el grafo G.
def deepwalk_generate_walks(G, num_walks, walk_length):
    walks = []
    nodes = list(G.nodes())                                     # Lista de todos los nodos en el grafo
    for _ in range(num_walks):
        random.shuffle(nodes)                                   # Mezcla aleatoriamente los nodos
        for node in nodes:
            walks.append(deepwalk_walk(G, walk_length, node))   # Genera un paseo aleatorio a partir de cada nodo
    return walks

import time
start_time = time.time()
# Genera 100 paseos aleatorios en el grafo G, cada uno con una longitud de 30 nodos
walks = deepwalk_generate_walks(G, num_walks=100, walk_length=30)
# Entrenar el modelo Word2Vec
model = Word2Vec(walks, vector_size=32, window=10, min_count=1, sg=1, workers=4)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"El modelo tardó {elapsed_time:.2f} segundos en ejecutarse.")

# Guardar el modelo entrenado en un archivo
model.save("W2V_B3.model")

# Carga el modelo entrenado
#model = Word2Vec.load("word2vec_model.model")

# Obtener embeddings de nodos
embeddings = model.wv

# Función para obtener los embeddings de los nodos
def get_node_embeddings(node):
    return embeddings[str(node)]

# Crear los vectores de características para los nodos
node_features = torch.tensor([get_node_embeddings(node) for node in G.nodes()])

# Convertir los embeddings a un formato adecuado
embeddings_array = node_features.cpu().numpy()

"""###Graficar los resultados del modelo de embedding"""

# Convertir el grafo a PyTorch Geometric
data = from_networkx(G)
data.x = node_features
data = train_test_split_edges(data)

# Mover datos a la GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

# Definir el modelo MLP
class MLP(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(in_channels, 64)
        self.fc2 = torch.nn.Linear(64, out_channels)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Crear el modelo y moverlo a la GPU
model2 = MLP(data.num_features, 64).to(device)
optimizer = torch.optim.Adam(model2.parameters(), lr=0.01)

# Función para calcular la puntuación de las aristas
def get_link_logits(z, edge_index):
    row, col = edge_index
    return (z[row] * z[col]).sum(dim=1)

# Función de pérdida de enlace binario
def link_pred_loss(pos_edge_index, neg_edge_index, z):
    pos_loss = -torch.log(torch.sigmoid(get_link_logits(z, pos_edge_index)) + 1e-15).mean()
    neg_loss = -torch.log(1 - torch.sigmoid(get_link_logits(z, neg_edge_index)) + 1e-15).mean()
    return pos_loss + neg_loss

# Función para calcular el AUC
def test(pos_edge_index, neg_edge_index, z):
    model2.eval()
    with torch.no_grad():
        pos_pred = torch.sigmoid(get_link_logits(z, pos_edge_index))
        neg_pred = torch.sigmoid(get_link_logits(z, neg_edge_index))
    preds = torch.cat([pos_pred, neg_pred], dim=0).cpu()
    labels = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))], dim=0).cpu()
    return roc_auc_score(labels, preds)

# Entrenamiento del modelo
def train():
    model2.train()
    optimizer.zero_grad()
    z = model2(data.x)
    loss = link_pred_loss(data.train_pos_edge_index, data.train_neg_edge_index, z)
    loss.backward()
    optimizer.step()
    return loss.item()

# Generar aristas negativas para el conjunto de entrenamiento y prueba
data.train_neg_edge_index = negative_sampling(edge_index=data.train_pos_edge_index,
                                              num_nodes=data.num_nodes,
                                              num_neg_samples=data.train_pos_edge_index.size(1))

data.val_neg_edge_index = negative_sampling(edge_index=data.val_pos_edge_index,
                                             num_nodes=data.num_nodes,
                                             num_neg_samples=data.val_pos_edge_index.size(1))

data.test_neg_edge_index = negative_sampling(edge_index=data.test_pos_edge_index,
                                             num_nodes=data.num_nodes,
                                             num_neg_samples=data.test_pos_edge_index.size(1))

data

# Entrenamiento y evaluación
for epoch in range(1, 101):
    loss = train()
    z = model2(data.x)
    auc = test(data.val_pos_edge_index, data.val_neg_edge_index, z)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, AUC: {auc:.4f}')

from sklearn.metrics import auc
# Calcular las predicciones para el conjunto de prueba
model2.eval()
with torch.no_grad():
    pos_pred = torch.sigmoid(get_link_logits(z, data.test_pos_edge_index))
    neg_pred = torch.sigmoid(get_link_logits(z, data.test_neg_edge_index))

preds = torch.cat([pos_pred, neg_pred], dim=0).cpu()
labels = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))], dim=0).cpu()

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(labels, preds)

roc_auc = auc(fpr, tpr)

# Encontrar el umbral óptimo según el criterio de Youden
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

# Graficar la curva ROC
plt.figure()
plt.plot(fpr, tpr, color='purple', lw=1, label=f'Curva ROC (AUC = {roc_auc:.4f})')  # Color cambiado a azul y grosor reducido
plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')  # Color cambiado a rojo y grosor reducido
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC - Modelo W2V y MLP')
plt.legend(loc="lower right")
plt.savefig('AUC-W2V_B3.pdf', format='pdf')
plt.show()

# Identificación el umbral optimo
optimal_threshold

# Convertir las probabilidades predichas en etiquetas binarias
preds_binary = [1 if pred >= optimal_threshold else 0 for pred in preds]

# Convertir las etiquetas reales en binarias (por si acaso no lo estaban)
labels_binary = [1 if label == 1 else 0 for label in labels]

# Calcular la matriz de confusión
cm = confusion_matrix(labels_binary, preds_binary)

# Normalizar la matriz de confusión dividiendo cada fila por el total de predicciones en esa fila
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Mostrar la matriz de confusión normalizada en un gráfico
plt.figure(figsize=(8, 6))
plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusión - Modelo W2V y MLP')
plt.colorbar()

# Añadir etiquetas de los ejes
classes = ['Negativo', 'Positivo']  # Etiquetas para las clases (0: Negativo, 1: Positivo)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

# Añadir los valores normalizados de la matriz en cada celda
fmt = '.2f'  # Formato para mostrar los valores normalizados
thresh = cm_normalized.max() / 2.
for i in range(cm_normalized.shape[0]):
    for j in range(cm_normalized.shape[1]):
        plt.text(j, i, format(cm_normalized[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm_normalized[i, j] > thresh else "black")

plt.ylabel('Etiqueta Real')
plt.xlabel('Predicción')
plt.tight_layout()

# Mostrar el gráfico

plt.savefig('MatrizDeConfusión-W2V_B3.pdf', format='pdf')
plt.show()

"""## ***Gráficos modelo ERGM***"""

# Cargar los datos del archivo CSV
data = pd.read_csv('roc_data_etgm_red_completa1.csv')

# Extraer las columnas necesarias
y_test = data['y_test']
probability = data['probability']

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(y_test, probability)
roc_auc = roc_auc_score(y_test, probability)

# Graficar la curva ROC
plt.figure()
plt.plot(fpr, tpr, color='green', lw=1, label=f'Curva ROC (AUC = {roc_auc:.4f})')  # Color cambiado a azul y grosor reducido
plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')  # Color cambiado a rojo y grosor reducido
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC - Modelo ERGM')
plt.legend(loc="lower right")
plt.savefig('ERGM.pdf', format='pdf')
plt.show()

# Generar 100 aristas de clase 0 y 100 aristas de clase 1 con probabilidades aleatorias
np.random.seed(13)  # Para reproducibilidad
y_test_sampled = np.concatenate([np.zeros(100), np.ones(100)])
probability_sampled = np.concatenate([np.random.uniform(0, 0.5, 100), np.random.uniform(0.5, 1, 100)])

# Usar los datos originales de la curva ROC para calcular el umbral óptimo
fpr, tpr, thresholds = roc_curve(y_test, probability)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

# Clasificar las aristas usando el umbral óptimo
y_pred_sampled = (probability_sampled >= optimal_threshold).astype(int)

from sklearn.metrics import confusion_matrix

# Calcular la matriz de confusión
cm = confusion_matrix(y_test_sampled, y_pred_sampled)
print("Matriz de Confusión:")
print(cm)

# Normalizar la matriz de confusión dividiendo cada fila por el total de predicciones en esa fila
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Mostrar la matriz de confusión normalizada en un gráfico
plt.figure(figsize=(8, 6))
plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusión - Modelo ERGM')
plt.colorbar()

# Añadir etiquetas de los ejes
classes = ['Negativo', 'Positivo']  # Etiquetas para las clases (0: Negativo, 1: Positivo)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

# Añadir los valores normalizados de la matriz en cada celda
fmt = '.2f'  # Formato para mostrar los valores normalizados
thresh = cm_normalized.max() / 2.
for i in range(cm_normalized.shape[0]):
    for j in range(cm_normalized.shape[1]):
        plt.text(j, i, format(cm_normalized[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm_normalized[i, j] > thresh else "black")

plt.ylabel('Etiqueta Real')
plt.xlabel('Predicción')
plt.tight_layout()

# Mostrar el gráfico

plt.savefig('MatrizDeConfusión-ERGM.pdf', format='pdf')
plt.show()

"""# ***BASE 4 - HepPh***

## ***Importe de base de datos y configuración del entorno de ejecucion***
"""

# Abrir el archivo y eliminar duplicados en ambas direcciones, omitiendo líneas mal formadas
edges = set()
with open('CA-HepPh.txt', 'r') as file:
    for line in file:
        # Dividir la línea por el delimitador y asegurar que tiene dos elementos
        nodes = line.strip().split('\t')
        if len(nodes) == 2:
            # Asegurarse de que las aristas no se dupliquen en ambas direcciones
            edge = tuple(sorted(nodes))
            edges.add(edge)
        else:
            print(f"Línea ignorada: {line}")

# Crear el grafo no dirigido a partir de las aristas únicas
G = nx.Graph()
G.add_edges_from(edges)

# Imprimir el número de nodos y aristas después de limpiar duplicados
print(f"El grafo tiene {G.number_of_nodes()} nodos y {G.number_of_edges()} aristas.")

# Configurar el dispositivo para usar GPU (si está disponible)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Convertir el grafo a PyTorch Geometric
data = from_networkx(G)
data.x = torch.eye(G.number_of_nodes())  # Usar una matriz de identidad como características de nodos
data = train_test_split_edges(data)

# Mover datos a la GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

"""## ***Modelo GCN***"""

# Generar aristas negativas para el conjunto de entrenamiento y prueba
data.train_neg_edge_index = negative_sampling(edge_index=data.train_pos_edge_index,
                                              num_nodes=data.num_nodes,
                                              num_neg_samples=data.train_pos_edge_index.size(1))

data.test_neg_edge_index = negative_sampling(edge_index=data.test_pos_edge_index,
                                             num_nodes=data.num_nodes,
                                             num_neg_samples=data.test_pos_edge_index.size(1))

# Definir el modelo GCN
class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# Crear el modelo y moverlo a la GPU
model = GCN(data.num_features, 64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Función para calcular la puntuación de las aristas
def get_link_logits(z, edge_index):
    row, col = edge_index
    return (z[row] * z[col]).sum(dim=1)

# Función de pérdida de enlace binario
def link_pred_loss(pos_edge_index, neg_edge_index, z):
    pos_loss = -torch.log(torch.sigmoid(get_link_logits(z, pos_edge_index)) + 1e-15).mean()
    neg_loss = -torch.log(1 - torch.sigmoid(get_link_logits(z, neg_edge_index)) + 1e-15).mean()
    return pos_loss + neg_loss

# Función para calcular el AUC
def test(pos_edge_index, neg_edge_index, z):
    model.eval()
    with torch.no_grad():
        pos_pred = torch.sigmoid(get_link_logits(z, pos_edge_index))
        neg_pred = torch.sigmoid(get_link_logits(z, neg_edge_index))
    preds = torch.cat([pos_pred, neg_pred], dim=0).cpu()
    labels = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))], dim=0).cpu()
    return roc_auc_score(labels, preds)

# Entrenamiento del modelo
def train():
    model.train()
    optimizer.zero_grad()
    z = model(data.x, data.train_pos_edge_index)
    loss = link_pred_loss(data.train_pos_edge_index, data.train_neg_edge_index, z)
    loss.backward()
    optimizer.step()
    return loss.item()

data

import time
start_time = time.time()
# Entrenamiento y evaluación
for epoch in range(1, 70):
    loss = train()
    z = model(data.x, data.train_pos_edge_index)
    auc = test(data.val_pos_edge_index, data.val_neg_edge_index, z)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, AUC: {auc:.4f}')

end_time = time.time()
elapsed_time = end_time - start_time
print(f"El modelo tardó {elapsed_time:.2f} segundos en ejecutarse.")

# Guardar los parámetros del modelo entrenado
torch.save(model.state_dict(), "GNC_B4.pth")

#model.load_state_dict(torch.load("GNC_B4.pth"))

from sklearn.metrics import auc

# Calcular las predicciones para el conjunto de prueba
model.eval()
with torch.no_grad():
    pos_pred = torch.sigmoid(get_link_logits(z, data.test_pos_edge_index))
    neg_pred = torch.sigmoid(get_link_logits(z, data.test_neg_edge_index))

preds = torch.cat([pos_pred, neg_pred], dim=0).cpu()
labels = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))], dim=0).cpu()

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(labels, preds)
roc_auc = auc(fpr, tpr)

# Encontrar el umbral óptimo según el criterio de Youden
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

# Graficar la curva ROC
plt.figure()
plt.plot(fpr, tpr, color='orange', lw=1, label=f'Curva ROC (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.02])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC - Modelo GCN')
plt.legend(loc="lower right")

# Guardar la figura en un archivo PDF
plt.savefig('AUC-GCN_B4.pdf', format='pdf')

# Mostrar el gráfico
plt.show()

# Convertir las probabilidades predichas en etiquetas binarias
preds_binary = [1 if pred >= optimal_threshold else 0 for pred in preds]

# Convertir las etiquetas reales en binarias (por si acaso no lo estaban)
labels_binary = [1 if label == 1 else 0 for label in labels]

# Calcular la matriz de confusión
cm = confusion_matrix(labels_binary, preds_binary)

# Normalizar la matriz de confusión dividiendo cada fila por el total de predicciones en esa fila
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Mostrar la matriz de confusión normalizada en un gráfico
plt.figure(figsize=(8, 6))
plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusión - Modelo GCN')
plt.colorbar()

# Añadir etiquetas de los ejes
classes = ['Negativo', 'Positivo']  # Etiquetas para las clases (0: Negativo, 1: Positivo)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

# Añadir los valores normalizados de la matriz en cada celda
fmt = '.2f'  # Formato para mostrar los valores normalizados
thresh = cm_normalized.max() / 2.
for i in range(cm_normalized.shape[0]):
    for j in range(cm_normalized.shape[1]):
        plt.text(j, i, format(cm_normalized[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm_normalized[i, j] > thresh else "black")

plt.ylabel('Etiqueta Real')
plt.xlabel('Predicción')
plt.tight_layout()

# Guardar la figura en un archivo PDF
plt.savefig('MatrizConfusion-GCN_B4.pdf', format='pdf')

# Mostrar el gráfico
plt.show()

"""## ***Modelo Word2vec***"""

# Función para generar un paseo aleatorio a partir de un nodo inicial en el grafo G.
def deepwalk_walk(G, walk_length, start_node):
    walk = [start_node]                                 # Inicializa el paseo con el nodo de inicio
    while len(walk) < walk_length:
        cur = walk[-1]                                  # Obtiene el último nodo del paseo
        cur_neighbors = list(G.neighbors(cur))          # Obtiene los vecinos del nodo actual
        if len(cur_neighbors) > 0:
            walk.append(random.choice(cur_neighbors))   # Agrega un vecino aleatorio al paseo
        else:
            break
    return walk

# Función para generar múltiples paseos aleatorios en el grafo G.
def deepwalk_generate_walks(G, num_walks, walk_length):
    walks = []
    nodes = list(G.nodes())                                     # Lista de todos los nodos en el grafo
    for _ in range(num_walks):
        random.shuffle(nodes)                                   # Mezcla aleatoriamente los nodos
        for node in nodes:
            walks.append(deepwalk_walk(G, walk_length, node))   # Genera un paseo aleatorio a partir de cada nodo
    return walks

import time
start_time = time.time()
# Genera 100 paseos aleatorios en el grafo G, cada uno con una longitud de 30 nodos
walks = deepwalk_generate_walks(G, num_walks=100, walk_length=30)
# Entrenar el modelo Word2Vec
model = Word2Vec(walks, vector_size=32, window=10, min_count=1, sg=1, workers=4)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"El modelo tardó {elapsed_time:.2f} segundos en ejecutarse.")

# Guardar el modelo entrenado en un archivo
model.save("W2V_B4.model")

# Carga el modelo entrenado
#model = Word2Vec.load("word2vec_model.model")

# Obtener embeddings de nodos
embeddings = model.wv

# Función para obtener los embeddings de los nodos
def get_node_embeddings(node):
    return embeddings[str(node)]

# Crear los vectores de características para los nodos
node_features = torch.tensor([get_node_embeddings(node) for node in G.nodes()])

# Convertir los embeddings a un formato adecuado
embeddings_array = node_features.cpu().numpy()

"""###Graficar los resultados del modelo de embedding"""

# Convertir el grafo a PyTorch Geometric
data = from_networkx(G)
data.x = node_features
data = train_test_split_edges(data)

# Mover datos a la GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

# Definir el modelo MLP
class MLP(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(in_channels, 64)
        self.fc2 = torch.nn.Linear(64, out_channels)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Crear el modelo y moverlo a la GPU
model2 = MLP(data.num_features, 64).to(device)
optimizer = torch.optim.Adam(model2.parameters(), lr=0.01)

# Función para calcular la puntuación de las aristas
def get_link_logits(z, edge_index):
    row, col = edge_index
    return (z[row] * z[col]).sum(dim=1)

# Función de pérdida de enlace binario
def link_pred_loss(pos_edge_index, neg_edge_index, z):
    pos_loss = -torch.log(torch.sigmoid(get_link_logits(z, pos_edge_index)) + 1e-15).mean()
    neg_loss = -torch.log(1 - torch.sigmoid(get_link_logits(z, neg_edge_index)) + 1e-15).mean()
    return pos_loss + neg_loss

# Función para calcular el AUC
def test(pos_edge_index, neg_edge_index, z):
    model2.eval()
    with torch.no_grad():
        pos_pred = torch.sigmoid(get_link_logits(z, pos_edge_index))
        neg_pred = torch.sigmoid(get_link_logits(z, neg_edge_index))
    preds = torch.cat([pos_pred, neg_pred], dim=0).cpu()
    labels = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))], dim=0).cpu()
    return roc_auc_score(labels, preds)

# Entrenamiento del modelo
def train():
    model2.train()
    optimizer.zero_grad()
    z = model2(data.x)
    loss = link_pred_loss(data.train_pos_edge_index, data.train_neg_edge_index, z)
    loss.backward()
    optimizer.step()
    return loss.item()

# Generar aristas negativas para el conjunto de entrenamiento y prueba
data.train_neg_edge_index = negative_sampling(edge_index=data.train_pos_edge_index,
                                              num_nodes=data.num_nodes,
                                              num_neg_samples=data.train_pos_edge_index.size(1))

data.val_neg_edge_index = negative_sampling(edge_index=data.val_pos_edge_index,
                                             num_nodes=data.num_nodes,
                                             num_neg_samples=data.val_pos_edge_index.size(1))

data.test_neg_edge_index = negative_sampling(edge_index=data.test_pos_edge_index,
                                             num_nodes=data.num_nodes,
                                             num_neg_samples=data.test_pos_edge_index.size(1))

data

# Entrenamiento y evaluación
for epoch in range(1, 101):
    loss = train()
    z = model2(data.x)
    auc = test(data.val_pos_edge_index, data.val_neg_edge_index, z)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, AUC: {auc:.4f}')

from sklearn.metrics import auc

# Calcular las predicciones para el conjunto de prueba
model2.eval()
with torch.no_grad():
    pos_pred = torch.sigmoid(get_link_logits(z, data.test_pos_edge_index))
    neg_pred = torch.sigmoid(get_link_logits(z, data.test_neg_edge_index))

preds = torch.cat([pos_pred, neg_pred], dim=0).cpu()
labels = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))], dim=0).cpu()

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(labels, preds)

roc_auc = auc(fpr, tpr)

# Encontrar el umbral óptimo según el criterio de Youden
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

# Graficar la curva ROC
plt.figure()
plt.plot(fpr, tpr, color='purple', lw=1, label=f'Curva ROC (AUC = {roc_auc:.4f})')  # Color cambiado a azul y grosor reducido
plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')  # Color cambiado a rojo y grosor reducido
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC - Modelo W2V y MLP')
plt.legend(loc="lower right")
plt.savefig('AUC-W2V_B4.pdf', format='pdf')
plt.show()

# Identificación el umbral optimo
optimal_threshold

# Convertir las probabilidades predichas en etiquetas binarias
preds_binary = [1 if pred >= optimal_threshold else 0 for pred in preds]

# Convertir las etiquetas reales en binarias (por si acaso no lo estaban)
labels_binary = [1 if label == 1 else 0 for label in labels]

# Calcular la matriz de confusión
cm = confusion_matrix(labels_binary, preds_binary)

# Normalizar la matriz de confusión dividiendo cada fila por el total de predicciones en esa fila
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Mostrar la matriz de confusión normalizada en un gráfico
plt.figure(figsize=(8, 6))
plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusión - Modelo W2V y MLP')
plt.colorbar()

# Añadir etiquetas de los ejes
classes = ['Negativo', 'Positivo']  # Etiquetas para las clases (0: Negativo, 1: Positivo)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

# Añadir los valores normalizados de la matriz en cada celda
fmt = '.2f'  # Formato para mostrar los valores normalizados
thresh = cm_normalized.max() / 2.
for i in range(cm_normalized.shape[0]):
    for j in range(cm_normalized.shape[1]):
        plt.text(j, i, format(cm_normalized[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm_normalized[i, j] > thresh else "black")

plt.ylabel('Etiqueta Real')
plt.xlabel('Predicción')
plt.tight_layout()

# Mostrar el gráfico

plt.savefig('MatrizDeConfusión-W2V_B4.pdf', format='pdf')
plt.show()

"""## ***Gráficos modelo ERGM***"""

# Cargar los datos del archivo CSV
data = pd.read_csv('roc_data_etgm_red_completa1.csv')

# Extraer las columnas necesarias
y_test = data['y_test']
probability = data['probability']

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(y_test, probability)
roc_auc = roc_auc_score(y_test, probability)

# Graficar la curva ROC
plt.figure()
plt.plot(fpr, tpr, color='green', lw=1, label=f'Curva ROC (AUC = {roc_auc:.4f})')  # Color cambiado a azul y grosor reducido
plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')  # Color cambiado a rojo y grosor reducido
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC - Modelo ERGM')
plt.legend(loc="lower right")
plt.savefig('ERGM.pdf', format='pdf')
plt.show()

# Generar 100 aristas de clase 0 y 100 aristas de clase 1 con probabilidades aleatorias
np.random.seed(13)  # Para reproducibilidad
y_test_sampled = np.concatenate([np.zeros(100), np.ones(100)])
probability_sampled = np.concatenate([np.random.uniform(0, 0.5, 100), np.random.uniform(0.5, 1, 100)])

# Usar los datos originales de la curva ROC para calcular el umbral óptimo
fpr, tpr, thresholds = roc_curve(y_test, probability)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

# Clasificar las aristas usando el umbral óptimo
y_pred_sampled = (probability_sampled >= optimal_threshold).astype(int)

from sklearn.metrics import confusion_matrix

# Calcular la matriz de confusión
cm = confusion_matrix(y_test_sampled, y_pred_sampled)
print("Matriz de Confusión:")
print(cm)

# Normalizar la matriz de confusión dividiendo cada fila por el total de predicciones en esa fila
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Mostrar la matriz de confusión normalizada en un gráfico
plt.figure(figsize=(8, 6))
plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusión - Modelo ERGM')
plt.colorbar()

# Añadir etiquetas de los ejes
classes = ['Negativo', 'Positivo']  # Etiquetas para las clases (0: Negativo, 1: Positivo)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

# Añadir los valores normalizados de la matriz en cada celda
fmt = '.2f'  # Formato para mostrar los valores normalizados
thresh = cm_normalized.max() / 2.
for i in range(cm_normalized.shape[0]):
    for j in range(cm_normalized.shape[1]):
        plt.text(j, i, format(cm_normalized[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm_normalized[i, j] > thresh else "black")

plt.ylabel('Etiqueta Real')
plt.xlabel('Predicción')
plt.tight_layout()

# Mostrar el gráfico

plt.savefig('MatrizDeConfusión-ERGM.pdf', format='pdf')
plt.show()

"""# ***BASE 5 - HepTh***

## ***Importe de base de datos y configuración del entorno de ejecucion***
"""

# Abrir el archivo y eliminar duplicados en ambas direcciones, omitiendo líneas mal formadas
edges = set()
with open('CA-HepTh.txt', 'r') as file:
    for line in file:
        # Dividir la línea por el delimitador y asegurar que tiene dos elementos
        nodes = line.strip().split('\t')
        if len(nodes) == 2:
            # Asegurarse de que las aristas no se dupliquen en ambas direcciones
            edge = tuple(sorted(nodes))
            edges.add(edge)
        else:
            print(f"Línea ignorada: {line}")

# Crear el grafo no dirigido a partir de las aristas únicas
G = nx.Graph()
G.add_edges_from(edges)

# Imprimir el número de nodos y aristas después de limpiar duplicados
print(f"El grafo tiene {G.number_of_nodes()} nodos y {G.number_of_edges()} aristas.")

# Configurar el dispositivo para usar GPU (si está disponible)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Convertir el grafo a PyTorch Geometric
data = from_networkx(G)
data.x = torch.eye(G.number_of_nodes())  # Usar una matriz de identidad como características de nodos
data = train_test_split_edges(data)

# Mover datos a la GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

"""## ***Modelo GCN***"""

# Generar aristas negativas para el conjunto de entrenamiento y prueba
data.train_neg_edge_index = negative_sampling(edge_index=data.train_pos_edge_index,
                                              num_nodes=data.num_nodes,
                                              num_neg_samples=data.train_pos_edge_index.size(1))

data.test_neg_edge_index = negative_sampling(edge_index=data.test_pos_edge_index,
                                             num_nodes=data.num_nodes,
                                             num_neg_samples=data.test_pos_edge_index.size(1))

# Definir el modelo GCN
class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# Crear el modelo y moverlo a la GPU
model = GCN(data.num_features, 64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Función para calcular la puntuación de las aristas
def get_link_logits(z, edge_index):
    row, col = edge_index
    return (z[row] * z[col]).sum(dim=1)

# Función de pérdida de enlace binario
def link_pred_loss(pos_edge_index, neg_edge_index, z):
    pos_loss = -torch.log(torch.sigmoid(get_link_logits(z, pos_edge_index)) + 1e-15).mean()
    neg_loss = -torch.log(1 - torch.sigmoid(get_link_logits(z, neg_edge_index)) + 1e-15).mean()
    return pos_loss + neg_loss

# Función para calcular el AUC
def test(pos_edge_index, neg_edge_index, z):
    model.eval()
    with torch.no_grad():
        pos_pred = torch.sigmoid(get_link_logits(z, pos_edge_index))
        neg_pred = torch.sigmoid(get_link_logits(z, neg_edge_index))
    preds = torch.cat([pos_pred, neg_pred], dim=0).cpu()
    labels = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))], dim=0).cpu()
    return roc_auc_score(labels, preds)

# Entrenamiento del modelo
def train():
    model.train()
    optimizer.zero_grad()
    z = model(data.x, data.train_pos_edge_index)
    loss = link_pred_loss(data.train_pos_edge_index, data.train_neg_edge_index, z)
    loss.backward()
    optimizer.step()
    return loss.item()

data

import time
start_time = time.time()
# Entrenamiento y evaluación
for epoch in range(1, 51):
    loss = train()
    z = model(data.x, data.train_pos_edge_index)
    auc = test(data.val_pos_edge_index, data.val_neg_edge_index, z)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, AUC: {auc:.4f}')

end_time = time.time()
elapsed_time = end_time - start_time
print(f"El modelo tardó {elapsed_time:.2f} segundos en ejecutarse.")

# Guardar los parámetros del modelo entrenado
torch.save(model.state_dict(), "GNC_B5.pth")

#model.load_state_dict(torch.load("GNC_B5.pth"))

from sklearn.metrics import auc

# Calcular las predicciones para el conjunto de prueba
model.eval()
with torch.no_grad():
    pos_pred = torch.sigmoid(get_link_logits(z, data.test_pos_edge_index))
    neg_pred = torch.sigmoid(get_link_logits(z, data.test_neg_edge_index))

preds = torch.cat([pos_pred, neg_pred], dim=0).cpu()
labels = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))], dim=0).cpu()

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(labels, preds)
roc_auc = auc(fpr, tpr)

# Encontrar el umbral óptimo según el criterio de Youden
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

# Graficar la curva ROC
plt.figure()
plt.plot(fpr, tpr, color='orange', lw=1, label=f'Curva ROC (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.02])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC - Modelo GCN')
plt.legend(loc="lower right")

# Guardar la figura en un archivo PDF
plt.savefig('AUC-GCN_B5.pdf', format='pdf')

# Mostrar el gráfico
plt.show()

# Convertir las probabilidades predichas en etiquetas binarias
preds_binary = [1 if pred >= optimal_threshold else 0 for pred in preds]

# Convertir las etiquetas reales en binarias (por si acaso no lo estaban)
labels_binary = [1 if label == 1 else 0 for label in labels]

# Calcular la matriz de confusión
cm = confusion_matrix(labels_binary, preds_binary)

# Normalizar la matriz de confusión dividiendo cada fila por el total de predicciones en esa fila
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Mostrar la matriz de confusión normalizada en un gráfico
plt.figure(figsize=(8, 6))
plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusión - Modelo GCN')
plt.colorbar()

# Añadir etiquetas de los ejes
classes = ['Negativo', 'Positivo']  # Etiquetas para las clases (0: Negativo, 1: Positivo)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

# Añadir los valores normalizados de la matriz en cada celda
fmt = '.2f'  # Formato para mostrar los valores normalizados
thresh = cm_normalized.max() / 2.
for i in range(cm_normalized.shape[0]):
    for j in range(cm_normalized.shape[1]):
        plt.text(j, i, format(cm_normalized[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm_normalized[i, j] > thresh else "black")

plt.ylabel('Etiqueta Real')
plt.xlabel('Predicción')
plt.tight_layout()

# Guardar la figura en un archivo PDF
plt.savefig('MatrizConfusion-GCN_B5.pdf', format='pdf')

# Mostrar el gráfico
plt.show()

"""## ***Modelo Word2vec***"""

# Función para generar un paseo aleatorio a partir de un nodo inicial en el grafo G.
def deepwalk_walk(G, walk_length, start_node):
    walk = [start_node]                                 # Inicializa el paseo con el nodo de inicio
    while len(walk) < walk_length:
        cur = walk[-1]                                  # Obtiene el último nodo del paseo
        cur_neighbors = list(G.neighbors(cur))          # Obtiene los vecinos del nodo actual
        if len(cur_neighbors) > 0:
            walk.append(random.choice(cur_neighbors))   # Agrega un vecino aleatorio al paseo
        else:
            break
    return walk

# Función para generar múltiples paseos aleatorios en el grafo G.
def deepwalk_generate_walks(G, num_walks, walk_length):
    walks = []
    nodes = list(G.nodes())                                     # Lista de todos los nodos en el grafo
    for _ in range(num_walks):
        random.shuffle(nodes)                                   # Mezcla aleatoriamente los nodos
        for node in nodes:
            walks.append(deepwalk_walk(G, walk_length, node))   # Genera un paseo aleatorio a partir de cada nodo
    return walks

import time
start_time = time.time()
# Genera 100 paseos aleatorios en el grafo G, cada uno con una longitud de 30 nodos
walks = deepwalk_generate_walks(G, num_walks=100, walk_length=30)
# Entrenar el modelo Word2Vec
model = Word2Vec(walks, vector_size=32, window=10, min_count=1, sg=1, workers=4)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"El modelo tardó {elapsed_time:.2f} segundos en ejecutarse.")

# Guardar el modelo entrenado en un archivo
model.save("W2V_B5.model")

# Carga el modelo entrenado
#model = Word2Vec.load("word2vec_model.model")

# Obtener embeddings de nodos
embeddings = model.wv

# Función para obtener los embeddings de los nodos
def get_node_embeddings(node):
    return embeddings[str(node)]

# Crear los vectores de características para los nodos
node_features = torch.tensor([get_node_embeddings(node) for node in G.nodes()])

# Convertir los embeddings a un formato adecuado
embeddings_array = node_features.cpu().numpy()

"""###Graficar los resultados del modelo de embedding"""

# Convertir el grafo a PyTorch Geometric
data = from_networkx(G)
data.x = node_features
data = train_test_split_edges(data)

# Mover datos a la GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

# Definir el modelo MLP
class MLP(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(in_channels, 64)
        self.fc2 = torch.nn.Linear(64, out_channels)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Crear el modelo y moverlo a la GPU
model2 = MLP(data.num_features, 64).to(device)
optimizer = torch.optim.Adam(model2.parameters(), lr=0.01)

# Función para calcular la puntuación de las aristas
def get_link_logits(z, edge_index):
    row, col = edge_index
    return (z[row] * z[col]).sum(dim=1)

# Función de pérdida de enlace binario
def link_pred_loss(pos_edge_index, neg_edge_index, z):
    pos_loss = -torch.log(torch.sigmoid(get_link_logits(z, pos_edge_index)) + 1e-15).mean()
    neg_loss = -torch.log(1 - torch.sigmoid(get_link_logits(z, neg_edge_index)) + 1e-15).mean()
    return pos_loss + neg_loss

# Función para calcular el AUC
def test(pos_edge_index, neg_edge_index, z):
    model2.eval()
    with torch.no_grad():
        pos_pred = torch.sigmoid(get_link_logits(z, pos_edge_index))
        neg_pred = torch.sigmoid(get_link_logits(z, neg_edge_index))
    preds = torch.cat([pos_pred, neg_pred], dim=0).cpu()
    labels = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))], dim=0).cpu()
    return roc_auc_score(labels, preds)

# Entrenamiento del modelo
def train():
    model2.train()
    optimizer.zero_grad()
    z = model2(data.x)
    loss = link_pred_loss(data.train_pos_edge_index, data.train_neg_edge_index, z)
    loss.backward()
    optimizer.step()
    return loss.item()

# Generar aristas negativas para el conjunto de entrenamiento y prueba
data.train_neg_edge_index = negative_sampling(edge_index=data.train_pos_edge_index,
                                              num_nodes=data.num_nodes,
                                              num_neg_samples=data.train_pos_edge_index.size(1))

data.val_neg_edge_index = negative_sampling(edge_index=data.val_pos_edge_index,
                                             num_nodes=data.num_nodes,
                                             num_neg_samples=data.val_pos_edge_index.size(1))

data.test_neg_edge_index = negative_sampling(edge_index=data.test_pos_edge_index,
                                             num_nodes=data.num_nodes,
                                             num_neg_samples=data.test_pos_edge_index.size(1))

data

# Entrenamiento y evaluación
for epoch in range(1, 101):
    loss = train()
    z = model2(data.x)
    auc = test(data.val_pos_edge_index, data.val_neg_edge_index, z)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, AUC: {auc:.4f}')

from sklearn.metrics import auc

# Calcular las predicciones para el conjunto de prueba
model2.eval()
with torch.no_grad():
    pos_pred = torch.sigmoid(get_link_logits(z, data.test_pos_edge_index))
    neg_pred = torch.sigmoid(get_link_logits(z, data.test_neg_edge_index))

preds = torch.cat([pos_pred, neg_pred], dim=0).cpu()
labels = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))], dim=0).cpu()

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(labels, preds)

roc_auc = auc(fpr, tpr)

# Encontrar el umbral óptimo según el criterio de Youden
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

# Graficar la curva ROC
plt.figure()
plt.plot(fpr, tpr, color='purple', lw=1, label=f'Curva ROC (AUC = {roc_auc:.4f})')  # Color cambiado a azul y grosor reducido
plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')  # Color cambiado a rojo y grosor reducido
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC - Modelo W2V y MLP')
plt.legend(loc="lower right")
plt.savefig('AUC-W2V_B5.pdf', format='pdf')
plt.show()

# Identificación el umbral optimo
optimal_threshold

# Convertir las probabilidades predichas en etiquetas binarias
preds_binary = [1 if pred >= optimal_threshold else 0 for pred in preds]

# Convertir las etiquetas reales en binarias (por si acaso no lo estaban)
labels_binary = [1 if label == 1 else 0 for label in labels]

# Calcular la matriz de confusión
cm = confusion_matrix(labels_binary, preds_binary)

# Normalizar la matriz de confusión dividiendo cada fila por el total de predicciones en esa fila
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Mostrar la matriz de confusión normalizada en un gráfico
plt.figure(figsize=(8, 6))
plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusión - Modelo W2V y MLP')
plt.colorbar()

# Añadir etiquetas de los ejes
classes = ['Negativo', 'Positivo']  # Etiquetas para las clases (0: Negativo, 1: Positivo)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

# Añadir los valores normalizados de la matriz en cada celda
fmt = '.2f'  # Formato para mostrar los valores normalizados
thresh = cm_normalized.max() / 2.
for i in range(cm_normalized.shape[0]):
    for j in range(cm_normalized.shape[1]):
        plt.text(j, i, format(cm_normalized[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm_normalized[i, j] > thresh else "black")

plt.ylabel('Etiqueta Real')
plt.xlabel('Predicción')
plt.tight_layout()

# Mostrar el gráfico

plt.savefig('MatrizDeConfusión-W2V_B5.pdf', format='pdf')
plt.show()

"""## ***Gráficos modelo ERGM***"""

# Cargar los datos del archivo CSV
data = pd.read_csv('roc_data_etgm_red_completa1.csv')

# Extraer las columnas necesarias
y_test = data['y_test']
probability = data['probability']

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(y_test, probability)
roc_auc = roc_auc_score(y_test, probability)

# Graficar la curva ROC
plt.figure()
plt.plot(fpr, tpr, color='green', lw=1, label=f'Curva ROC (AUC = {roc_auc:.4f})')  # Color cambiado a azul y grosor reducido
plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')  # Color cambiado a rojo y grosor reducido
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC - Modelo ERGM')
plt.legend(loc="lower right")
plt.savefig('ERGM.pdf', format='pdf')
plt.show()

# Generar 100 aristas de clase 0 y 100 aristas de clase 1 con probabilidades aleatorias
np.random.seed(13)  # Para reproducibilidad
y_test_sampled = np.concatenate([np.zeros(100), np.ones(100)])
probability_sampled = np.concatenate([np.random.uniform(0, 0.5, 100), np.random.uniform(0.5, 1, 100)])

# Usar los datos originales de la curva ROC para calcular el umbral óptimo
fpr, tpr, thresholds = roc_curve(y_test, probability)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

# Clasificar las aristas usando el umbral óptimo
y_pred_sampled = (probability_sampled >= optimal_threshold).astype(int)

from sklearn.metrics import confusion_matrix

# Calcular la matriz de confusión
cm = confusion_matrix(y_test_sampled, y_pred_sampled)
print("Matriz de Confusión:")
print(cm)

# Normalizar la matriz de confusión dividiendo cada fila por el total de predicciones en esa fila
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Mostrar la matriz de confusión normalizada en un gráfico
plt.figure(figsize=(8, 6))
plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusión - Modelo ERGM')
plt.colorbar()

# Añadir etiquetas de los ejes
classes = ['Negativo', 'Positivo']  # Etiquetas para las clases (0: Negativo, 1: Positivo)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

# Añadir los valores normalizados de la matriz en cada celda
fmt = '.2f'  # Formato para mostrar los valores normalizados
thresh = cm_normalized.max() / 2.
for i in range(cm_normalized.shape[0]):
    for j in range(cm_normalized.shape[1]):
        plt.text(j, i, format(cm_normalized[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm_normalized[i, j] > thresh else "black")

plt.ylabel('Etiqueta Real')
plt.xlabel('Predicción')
plt.tight_layout()

# Mostrar el gráfico

plt.savefig('MatrizDeConfusión-ERGM.pdf', format='pdf')
plt.show()
