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
  return count

def knn_leave_one_out(method, k, C):
  # Supprimer toutes les classe > 5   pour qst 4.2
  if C == 6:
    for x, y in list(alldata.approche[method].keys()):
      if x >= 6:
        del alldata.approche[method][x, y]

  true_labels = {}
  pred_labels = {}
  for (test_classe, test_ech) in alldata.approche[method]:
    test_vec = alldata.approche[method][(test_classe, test_ech)]
    train_db = copy.deepcopy(alldata.approche[method])
    train_db.pop((test_classe, test_ech))
    count = knn_classify(train_db, test_vec, k)
    pred_classe = max(count, key=count.get)

    true_labels[test_classe,test_ech] = test_classe
    pred_labels[test_classe,test_ech] = pred_classe
  return true_labels, pred_labels
