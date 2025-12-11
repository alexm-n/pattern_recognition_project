import numpy as np
from matplotlib import pyplot as plt
import all_vectors
import k_means_v2 as kmean
import knn
import vote_maj


def labels_to_tp(true_labels, pred_labels):
    tp = {c: 0 for c in range(1, 10)}
    fp = {(p, t): 0 for p in range(1, 10) for t in range(1, 10)}
    fn = {(t, p): 0 for t in range(1, 10) for p in range(1, 10)}
    for x in range(1, 10):
        for y in range(1, 12):
            if true_labels[x, y] == pred_labels[x, y]:
                tp[true_labels[x, y]] += 1
            else:
                fn[(true_labels[x, y], pred_labels[x, y])] += 1
                fp[(pred_labels[x, y], true_labels[x, y])] += 1

    return tp, fp, fn


def rappel(tp, fn):
    r = {}
    r_glob = 0
    print(tp)
    print(fn)
    for k in tp:
        fn_k = 0
        for d in range(1, 10):
            if k != d:
                fn_k += fn[k, d]
        if fn_k + tp[k] == 0:
            r[k] = 0
        else:
            r[k] = tp[k] / (tp[k] + fn_k)
        r_glob += r[k]
    r_glob = r_glob / len(tp)
    return r, r_glob


def precision(tp, fp):
    r = {}
    r_glob = 0
    for k in tp:
        fp_k = 0
        for d in range(1, 10):
            if k != d:
                fp_k += fp[k, d]
        if fp_k + tp[k] == 0:
            r[k] = 0
        else:
            r[k] = tp[k] / (tp[k] + fp_k)
        r_glob += r[k]
    r_glob = r_glob / len(tp)
    return r, r_glob


def fscore(rap, prec, beta):
    return ((1 + beta ** 2) * prec * rap) / ((beta ** 2) * prec + rap)


def conf_mat(tp, fp, fn):
    matrice = [[0 for i in range(9)] for i in range(9)]
    for i in range(1, 10):
        matrice[i - 1][i - 1] = tp[i]
        for x in range(i + 1, 10):
            matrice[x - 1][i - 1] = fp[x, i]
            matrice[i - 1][x - 1] = fn[i, x]
    return matrice


def dessin_mat(mat):
    plt.imshow(mat, cmap="Blues")
    plt.colorbar()
    # Ajouter les valeurs dans les cases
    for x in range(9):
        for y in range(9):
            plt.text(y, x, mat[x][y], ha='center', va='center', color='black')

    plt.xlabel("Estimation")
    plt.ylabel("Référence")
    plt.title("Matrice de confusion")
    plt.show()



def prerap_curve(true_labels_dict, pred_labels_dict, k):
    for cls in range(1, 10):
        pre_tot = []
        rap_tot = []
        n = 0
        x = np.arange(0, 1, 0.1)
        for y in range(1, k + 1):
            if true_labels_dict[cls, y] == pred_labels_dict[cls, y]:
                n += 1
            pre_tot.append(n / y)
            rap_tot.append(n / 11)

        plt.plot(x, pre_tot, marker='o', label='precision')
        plt.plot(x, rap_tot, marker='o', label='rappel')
        plt.title(f"per/rap courbe de classe {cls}")
        plt.legend()
        plt.grid(True)
        plt.show()


C = 10  #nombre de classe 10/6
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
# for k in [5,9,13]:
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

# #Vote Majoritaire
# true_labels, pred_labels = vote_maj.vm_leave_one_out(4, C)
# tp, fp, fn = labels_to_tp(true_labels,pred_labels)
# tab_rap , rapp = rappel(tp,fn)
# tab_prc,prcc = precision(tp,fp)
# fs = fscore(rapp,prcc,1)
# mat = conf_mat(tp, fp, fn)
# dessin_mat(mat)
# prerap_curve(true_labels,pred_labels,k=10)
# print(f"rappel {rapp}, precision {prcc}, F1-score {fs}")
