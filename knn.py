import os
import math
import copy
import all_vectors as alldata

def euclidean_dist(v1, v2):
  s = 0.0
  for i in range(len(v1)):
    s+= (v1[i] - v2[i]) ** 2
  return math.sqrt(s)

def knn_classify(train_db, query_vec, k):
  distances = []
  for (classe, ech) in train_db:
    distances.append((euclidean_dist(query_vec, train_db[classe, ech]), classe))
  distances.sort(key=lambda x: x[0])
  k_nearest = distances[:k]
  count = {}
  for _, classe in k_nearest:
    count[classe] = count.get(classe, 0) + 1
  return max(count, key=count.get)

if __name__ == "__main__":
  method = "E34"
  train_db = copy.deepcopy(alldata.approche[method])
  test_classe = 2
  test_ech = 3
  test_vec = alldata.approche[method][(test_classe, test_ech)]
  train_db.pop((test_classe, test_ech))
  print(knn_classify(train_db, test_vec, 3))
