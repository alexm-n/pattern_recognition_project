import math

import numpy as np
from matplotlib import pyplot as plt
import all_vectors
import k_means_v2 as kmean
import knn_vf as knn
import vote_maj


def labels_to_tp(true_labels, pred_labels):#transformer les labels en tp,fp,fn pour les metriques et la matrice
    tp = {c: 0 for c in range(1, C)}
    fp = {(p, t): 0 for p in range(1, C) for t in range(1, C)}
    fn = {(t, p): 0 for t in range(1, C) for p in range(1, C)}
    for x in range(1, C):
        for y in range(1, 12):
            if true_labels[x, y] == pred_labels[x, y]:#si la classe presite est la bonne on rajout a tp de la classe
                tp[true_labels[x, y]] += 1
            else:
                fn[(true_labels[x, y], pred_labels[x, y])] += 1  #designe que elle apartient a true classe mais assigne a la classe pred classe
                fp[(pred_labels[x, y], true_labels[x, y])] += 1  #designe que elle apartient a pred_classe mais assigne a la classe true classe

    return tp, fp, fn


def rappel(tp, fn):
    r = {}
    r_glob = 0
    for k in tp:
        fn_k = 0
        for d in range(1, C):
            if k != d:
                fn_k += fn[k, d]
        if fn_k + tp[k] == 0:
            r[k] = 0
        else:
            r[k] = tp[k] / (tp[k] + fn_k)
        r_glob += r[k]
    r_glob = r_glob / len(tp)
    return r, r_glob #on a le rappel en tableau de chaque classe et le rappel global


def precision(tp, fp):
    r = {}
    r_glob = 0
    for k in tp:
        fp_k = 0
        for d in range(1, C):
            if k != d:
                fp_k += fp[k, d]
        if fp_k + tp[k] == 0:
            r[k] = 0
        else:
            r[k] = tp[k] / (tp[k] + fp_k)
        r_glob += r[k]
    r_glob = r_glob / len(tp)
    return r, r_glob #on a la precision en tableau de chaque classe et la precision globale


def fscore(rap, prec, beta):
    return ((1 + beta ** 2) * prec * rap) / ((beta ** 2) * prec + rap)


def conf_mat(tp, fp, fn):
    matrice = [[0 for i in range(C-1)] for i in range(C-1)]
    for i in range(1, C):
        matrice[i - 1][i - 1] = tp[i]
        for x in range(i + 1, C):
            matrice[x - 1][i - 1] = fp[i,x] #s'occupe de la colonne
            matrice[i - 1][x - 1] = fn[i, x] #s'occupe de la ligne
    return matrice


def dessin_mat(mat):
    plt.imshow(mat, cmap="Blues")
    plt.colorbar()
    # Ajouter les valeurs dans les cases
    for x in range(C-1):
        for y in range(C-1):
            plt.text(y, x, mat[x][y], ha='center', va='center', color='black')

    plt.xlabel("Estimation")
    plt.ylabel("Référence")
    plt.title("Matrice de confusion")
    plt.show()



def prerap_curve(true_labels_dict, pred_labels_dict, k):
    max_row = 5
    n_rows = math.ceil(C / max_row)
    n_cols = min(C, max_row)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows)) #permet de toute les figures mettre dans 1 ou 2 lignes
    axes = axes.flatten()

    for cls in range(1, C):
        pre_tot = []
        rap_tot = []
        n = 0
        x = np.arange(0, 1, 0.1)
        for y in range(1, k + 1):
            if true_labels_dict[cls, y] == pred_labels_dict[cls, y]:
                n += 1  #calcule le nombre d'elements bien classifié
            pre_tot.append(n / y)
            rap_tot.append(n / 11)

        ax = axes[cls - 1]
        ax.plot(x, pre_tot, marker='o', label='Precision')
        ax.plot(x, rap_tot, marker='o', label='Rappel')
        ax.plot(rap_tot, pre_tot, marker='o', label='prec_rap courbe')
        auc_pr = np.trapezoid(pre_tot, rap_tot) #calculer l'AUC de la courbe precision/rappel
        ax.set_title(f"Classe {cls}\nAUC-PR = {auc_pr:.3f}")
        ax.grid(True)
        ax.legend()  #dessine 3 courbes precision, rappel et precision/rappel

    plt.show() #affiche les courbes de toutes les classes


C = 10  #nombre de classe soit 10/6
print("evaluation")

##KNN
# for mtd in all_vectors.approche:
#     true_labels, pred_labels = knn.knn_leave_one_out(mtd, 4, C)
#     tp, fp, fn = labels_to_tp(true_labels,pred_labels)
#     tab_rap , rapp = rappel(tp,fn)
#     tab_prc,prcc= precision(tp,fp)
#     fs = fscore(rapp,prcc,1)
#     mat = conf_mat(tp, fp, fn)
#     dessin_mat(mat)
#     prerap_curve(true_labels,pred_labels,k=10)
#     print(f"rappel {rapp}, precision {prcc}, F1-score {fs}")

##Kmean
# for k in [5,9,13]:  #test des valeurs de k en 5, 9 et 13
#     for mtd in all_vectors.approche:
#         true_labels, pred_labels = kmean.kmeans_classify(mtd,k)
#         tp, fp, fn = labels_to_tp(true_labels,pred_labels)
#         tab_rap , rapp = rappel(tp,fn)
#         tab_prc,prcc = precision(tp,fp)
#         fs = fscore(rapp,prcc,1)
#         mat=conf_mat(tp,fp,fn)
#         dessin_mat(mat)
#         prerap_curve(true_labels,pred_labels,k=10)
#         print(f"rappel {rapp}, precision {prcc}, F1-score {fs}")

#Vote Majoritaire
# true_labels, pred_labels = vote_maj.vm_leave_one_out(4, C)
# tp, fp, fn = labels_to_tp(true_labels,pred_labels)
# tab_rap , rapp = rappel(tp,fn)
# tab_prc,prcc = precision(tp,fp)
# fs = fscore(rapp,prcc,1)
# mat = conf_mat(tp, fp, fn)
# dessin_mat(mat)
# prerap_curve(true_labels,pred_labels,k=10)
# print(f"rappel {rapp}, precision {prcc}, F1-score {fs}")

