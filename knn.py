import os
import math
import copy
import matplotlib.pyplot as plt
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

def leave_one_out(database, method, k):
    true_labels = []
    pred_labels = []
    for test_classe, test_ech in list(database.keys()):
        test_vec = train_db[(test_classe, test_ech)]
        saved = train_db.pop((test_classe, test_ech))
        pred_classe = knn_classify(train_db, test_vec, k)
        true_labels.append(test_classe)
        pred_labels.append(pred_classe)
        train_db[(test_classe, test_ech)] = saved
    return true_labels, pred_labels

def get_f_score(precision, rappel, beta=1):
    precision = float(precision)
    rappel = float(rappel)
    if precision + rappel != 0:
        return (1 + beta ** 2) * (precision * rappel / ((beta ** 2) * precision + rappel))
    else:
        return 0

def metriques(true_labels, pred_labels, nb_classes=9, beta=1):
    conf_matrix = [[0 for _ in range(nb_classes)] for _ in range(nb_classes)]
    tp = {}
    fp = {}
    fn = {}
    precision = {}
    rappel = {}
    f_score = {}
    for classe in [i for i in range(1,10)]:
        tp[classe] = 0
        fp[classe] = 0
        fn[classe] = 0
        precision[classe] = 0
        rappel[classe] = 0
        f_score[classe] = 0
    glob_precision = 0
    glob_rappel = 0
    glob_f_score = 0
    for true_classe, pred_classe in zip(true_labels, pred_labels):
        conf_matrix[true_classe-1][pred_classe-1] += 1
        tp[true_classe] = tp.get(true_classe, 0) + (true_classe == pred_classe)
        fp[true_classe] = fp.get(true_classe, 0) + (true_classe != pred_classe)
        fn[pred_classe] = fn.get(pred_classe, 0) + (true_classe != pred_classe)
    for classe in [i for i in range(1, nb_classes + 1)]:
        precision[classe] = tp[classe] / (tp[classe] + fp[classe])
        rappel[classe] = tp[classe] / (tp[classe] + fn[classe])
        f_score[classe] = get_f_score(precision[classe], rappel[classe], beta)
        glob_precision += precision[classe]
        glob_rappel += rappel[classe]
    glob_precision /= nb_classes
    glob_rappel /= nb_classes
    glob_f_score = get_f_score(glob_precision, glob_rappel, beta)
    return conf_matrix, precision, rappel, f_score, glob_precision, glob_rappel, glob_f_score

def print_matrix(matrix):
    print("[", end="")
    for i in range(len(matrix)):
        if i == 0:
            print("[", end="")
        else:
            print(" [", end="")
        for j in range(len(matrix[i])):
            print(matrix[i][j], end="")
            if j == len(matrix[i])-1:
                if i == len(matrix)-1:
                    print("]]")
                else:
                    print("],")
            else:
                print(", ", end="")

if __name__ == "__main__":
  k = 10
  nb_classes = 9
  method = "E34"
  train_db = copy.deepcopy(alldata.approche[method])
  true1, pred1 = leave_one_out(train_db, method, k)
  cm1, precision, rappel, f_score, glob_precision, glob_rappel, glob_f_score = metriques(true1, pred1, beta=2)
  print_matrix(cm1)
  print(glob_precision)
  print(glob_rappel)
  print(glob_f_score)
