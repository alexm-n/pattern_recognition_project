import math
import copy

from matplotlib import pyplot as plt

import all_vectors as alldata


def euclidean_dist(v1, v2):#calcule la distance euclidienne
    s = 0.0
    for i in range(len(v1)):
        s += (v1[i] - v2[i]) ** 2
    return math.sqrt(s)


def knn_classify(train_db, query_vec, k):
    distances = []
    for (classe, ech) in train_db:
        distances.append((euclidean_dist(query_vec, train_db[classe, ech]), classe))
        #pour chaque point dans train_db on calcule la distance au vecteur tes
    distances.sort(key=lambda x: x[0]) #on les trie du plus petit au plus grand
    k_nearest = distances[:k]  #on choisit les k plus proche
    count = {}
    for _, classe in k_nearest:
        count[classe] = count.get(classe, 0) + 1
    return count


def knn_leave_one_out(method, k, C):
    print(f"\n------------- Pour {method} -----------------")
    # Supprimer toutes les classe > 5   pour qst 4.2
    if C == 6:
        for x, y in list(alldata.approche[method].keys()):
            if x >= 6:
                del alldata.approche[method][x, y]

    true_labels = {}
    pred_labels = {}
    for (test_classe, test_ech) in alldata.approche[method]:# on prend chaque point comme test
        test_vec = alldata.approche[method][(test_classe, test_ech)]
        train_db = copy.deepcopy(alldata.approche[method])
        train_db.pop((test_classe, test_ech))
        count = knn_classify(train_db, test_vec, k)
        pred_classe = max(count, key=count.get)

        true_labels[test_classe, test_ech] = test_classe
        pred_labels[test_classe, test_ech] = pred_classe
    return true_labels, pred_labels


#partie pour courbe precision/rappel facon generale
def knn_predict_proba(train_db, query_vec, k, nb_classes=9):
    distances = []
    for (classe, ech) in train_db:
        distances.append((euclidean_dist(query_vec, train_db[(classe, ech)]), classe))
    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:k]

    count = {c: 0 for c in range(1, nb_classes + 1)}
    for _, classe in k_nearest:
        count[classe] += 1

    return [count[c] / k for c in range(1, nb_classes + 1)] #la meme maniere que knn_predict mais on divise par k

def leave_one_out_proba(database, k, nb_classes=9):
    true_labels = []
    proba_labels = []

    for (test_classe, test_ech) in list(database.keys()):
        test_vec = database[(test_classe, test_ech)]
        train_db = copy.deepcopy(database)
        train_db.pop((test_classe, test_ech))

        proba = knn_predict_proba(train_db, test_vec, k, nb_classes)

        true_labels.append(test_classe)
        proba_labels.append(proba)

    return true_labels, proba_labels  #on a les labels true et proba

def print_prc( k, method, nb_classes):
    if nb_classes==10:
        database = copy.deepcopy(alldata.approche[method])
    else:
        database = copy.deepcopy(alldata.approche[method])
        for classe in range(6, 10):
            for ech in range(1, 12):
                database.pop((classe, ech))
    true_labels, proba_labels = leave_one_out_proba(database, k, nb_classes)

    plt.figure(figsize=(8, 6))
    for classe in range(1, nb_classes + 1):
        y_true = [1 if t == classe else 0 for t in true_labels]
        y_score = [p[classe - 1] for p in proba_labels]  #la proba de la classe

        precision, recall = precision_recall_calc(y_true, y_score)
        auc = auc_calc(precision, recall) #calcule de l'auc de la courbe

        plt.plot(recall, precision, label=f"Classe {classe} (AUC={auc:.2f})") #dessine pour chaque claase sa courbe

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Courbes Precision-Recall KNN pour {method}")
    plt.legend()
    plt.show()

def precision_recall_calc(y_true, y_score):
    # y_true : 1 pour positif, 0 pour négatif
    # y_score : score/probabilité
    pairs = list(zip(y_score, y_true))
    pairs.sort(reverse=True)
    tp = 0
    fp = 0
    fn = sum(y_true)
    precisions = []
    recalls = []
    for score, label in pairs:
        if label == 1:
            tp += 1
            fn -= 1
        else:
            fp += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precisions.append(precision)
        recalls.append(recall)
    return precisions, recalls

def get_f_score(precision, rappel, beta=1):
    if precision + rappel != 0:
        return (1 + beta ** 2) * (precision * rappel / ((beta ** 2) * precision + rappel))
    else:
        return 0

def metriques(true_labels, pred_labels, nb_classes=9, beta=1): #les memes metriques dans le fichier evaluation
    conf_matrix = [[0 for _ in range(nb_classes)] for _ in range(nb_classes)]
    tp = {}
    fp = {}
    fn = {}
    precision = {}
    rappel = {}
    f_score = {}
    for classe in [i for i in range(1, nb_classes + 1)]:
        tp[classe] = 0
        fp[classe] = 0
        fn[classe] = 0
        precision[classe] = 0
        rappel[classe] = 0
        f_score[classe] = 0
    glob_precision = 0
    glob_rappel = 0
    for true_classe, pred_classe in zip(true_labels, pred_labels):
        conf_matrix[true_classe-1][pred_classe-1] += 1
        for classe in range(1, nb_classes + 1):
            if true_classe == classe and pred_classe == classe:
                tp[classe] += 1
            elif true_classe != classe and pred_classe == classe:
                fp[classe] += 1
            elif true_classe == classe and pred_classe != classe:
                fn[classe] += 1
    for classe in [i for i in range(1, nb_classes + 1)]:
        if tp[classe] + fp[classe] == 0:
            precision[classe] = 0
        else:
            precision[classe] = tp[classe] / (tp[classe] + fp[classe])
        if tp[classe] + fn[classe] == 0:
            rappel[classe] = 0
        else:
            rappel[classe] = tp[classe] / (tp[classe] + fn[classe])
        f_score[classe] = get_f_score(precision[classe], rappel[classe], beta)
        glob_precision += precision[classe]
        glob_rappel += rappel[classe]
    glob_precision /= nb_classes
    glob_rappel /= nb_classes
    glob_f_score = get_f_score(glob_precision, glob_rappel, beta)
    return conf_matrix, precision, rappel, f_score, glob_precision, glob_rappel, glob_f_score

def auc_calc(precision, recall):
    pairs = sorted(zip(recall, precision))
    recall_sorted, precision_sorted = zip(*pairs)
    auc = 0.0
    for i in range(1, len(recall_sorted)):
        delta_r = recall_sorted[i] - recall_sorted[i-1]
        avg_p = (precision_sorted[i] + precision_sorted[i-1]) / 2
        auc += delta_r * avg_p
    return auc
