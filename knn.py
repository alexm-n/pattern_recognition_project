import math
import os

def euclidean_dist(v1, v2):
  s = 0.0
  for i in range(len(v1)):
    s+= (v1[i] - v2[i]) ** 2
  return math.sqrt(s)

def load_vector(filepath):
  with open(filepath, "r") as f:
    return [float(line) for line in f]

def load_database(folderpath):
  database = []
  for filename in os.listdir(folderpath):
    nclasse = int(filename[1:3])
    filepath = os.path.join(folderpath, filename)
    vec = load_vector(filepath)
    database.append((vec, nclasse))
  return database

def knn_classify(database, query_vec, k):
  distances = [(euclidean_dist(query_vec, data_vec), nclasse) for (data_vec, nclasse) in database]
  distances.sort(key=lambda x: x[0])
  k_nearest = distances[:k]
  count = {}
  for _, nclasse in k_nearest:
    count[nclasse] = count.get(nclasse, 0) + 1
  return max(count, key=count.get)

if __name__ == "__main__":
  filepath = "./data/F0/"
  database = load_database(filepath)
  qvec = load_vector("./data/s04n003.F0")
  print(knn_classify(database, qvec, 3))
