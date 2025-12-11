import copy
import os
from knn import  knn_classify
import all_vectors
import random



def max_iter(p):
    bast = {}
    for x in p:
        bast[x] =bast.get(x, 0) + 1
    max= 0
    bonne_classe = []
    for x,y in bast.items():
        if max <= y:
            max = y
            bonne_classe.append(x)
    rd = random.randint(0,len(bonne_classe)-1)
    return bonne_classe[rd]

def vm_leave_one_out(k):
    true_labels = {}
    pred_labels = {}
    proba_labels = {}

    for test_classe in range(1,10):
        for test_ech in range(1,12):
            pred_class = []
            prob_class = []
            for method in all_vectors.approche:
                # Supprimer toutes les classe > 5   pour qst 4.2
               # for x, y in list(alldata.approche[method].keys()):
               #   if x >= 6:
               #     del alldata.approche[method][x, y]
                
                test_vec = all_vectors.approche[method][(test_classe, test_ech)]
                train_db = copy.deepcopy(all_vectors.approche[method])
                train_db.pop((test_classe, test_ech))
                count = knn_classify(train_db, test_vec, k)
                pred_class.append(max(count, key=count.get))
                prob_class.append([count.get(c, 0) / k for c in range(1, 10)])

            true_labels[test_classe, test_ech] = test_classe
            pred_labels[test_classe, test_ech] = max_iter(pred_class)
            proba_labels[test_classe, test_ech] = prob_class[pred_class.index(max_iter(pred_class))]



    return true_labels, pred_labels, proba_labels

true_labels, pred_labels, proba_labels = vm_leave_one_out(4)
print(true_labels)
