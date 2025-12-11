import copy
from knn_vf import knn_classify
import all_vectors
import random


def max_iter(p):
    bast = {}
    for x in p:
        bast[x] = bast.get(x, 0) + 1  #on note le nombre de fois qu'une classe est citée
    max = 0
    bonne_classe = []
    for x, y in bast.items():
        if max < y: #si une classe est cite plus que max elle prend ca place
            max = y
            bonne_classe = [x]
        elif max==y:#si max est le meme que le precedant alors les ajouter les 2
            bonne_classe.append(x)
    rd = random.randint(0, len(bonne_classe) - 1)  #dans le cas de classes egales on choisit au hasard
    return bonne_classe[rd]


def vm_leave_one_out(k, C):
    true_labels = {}
    pred_labels = {}

    for test_classe in range(1, C):
        for test_ech in range(1, 12):
            pred_class = []
            for method in all_vectors.approche: #pour 1 point le faire pour chaque method
                if C == 6: # pour la qst 4.2 seulement 5 classes
                    for x, y in list(all_vectors.approche[method].keys()):
                        if x >= 6:
                            del all_vectors.approche[method][x, y]

                test_vec = all_vectors.approche[method][(test_classe, test_ech)]  #prend les valeurs du vecteur test
                train_db = copy.deepcopy(all_vectors.approche[method])
                train_db.pop((test_classe, test_ech))  #on enleve le point test
                count = knn_classify(train_db, test_vec, k)
                pred_class.append(max(count, key=count.get)) #la classe predite par une des methodes

            true_labels[test_classe, test_ech] = test_classe
            pred_labels[test_classe, test_ech] = max_iter(pred_class)  #on choisit une des classes des methodes

    return true_labels, pred_labels


