from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve,auc,precision_recall_curve

import knn
import vote_maj
import k_means_v2 as kk


def labels_to_tp(true_labels, pred_labels):
    tp = {c: 0 for c in range(1, C)}
    fp = {(p, t): 0 for p in range(1, C) for t in range(1, C)}
    fn = {(t, p): 0 for t in range(1, C) for p in range(1, C)}
    for x in range(1,C):
        for y in range(1,12):
            if true_labels[x,y] == pred_labels[x,y]:
                tp[true_labels[x,y]] += 1
            else:
                fn[(true_labels[x,y],pred_labels[x,y])] += 1
                fp[(pred_labels[x,y], true_labels[x,y])] += 1

    return tp, fp, fn


def rappel(tp, fn):
    r = {}
    r_glob = 0
    print(tp)
    print(fn)
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
    return r, r_glob


def precision(tp, fp):
    r = {}
    r_glob = 0
    for k in tp:
        fp_k = 0
        for d in range(1, C):
            if k!=d:
                fp_k +=  fp[k,d]
        if fp_k+tp[k] == 0:
            r[k] = 0
        else:
            r[k] = tp[k] / (tp[k] + fp_k)
        r_glob += r[k]
    r_glob = r_glob / len(tp)
    return r, r_glob


def fscore(rap,prec,beta):
    return ((1+beta**2)*prec*rap)/((beta**2)*prec+rap)


def conf_mat(tp,fp,fn):
    matrice = [[0 for i in range(C-1)] for i in range(C-1)]
    for i in range(1,C):
        matrice[i-1][i-1] = tp[i]
        for x in range(i+1,C):
            matrice[x-1][i-1]=fp[x,i]
            matrice[i-1][x-1]=fn[i,x]
    return matrice

def dessin_mat(mat):
    plt.imshow(mat, cmap="Blues")
    plt.colorbar()

    # Ajouter les valeurs dans les cases
    for x in range(C-1):
        for y in range(C-1):
            plt.text(y,x, mat[x][y], ha='center', va='center', color='black')

    plt.xlabel("Estimation")
    plt.ylabel("Référence")
    plt.title("Matrice de confusion")
    plt.show()



def prerap_curve(true_labels_dict, pred_labels_dict,k=10):
    for cls in range(1, C):
        pre_tot = []
        rap_tot = []
        n=0
        x = np.arange(0, 1, 0.1)
        for y in range(1,k+1):
            if true_labels_dict[cls, y] == pred_labels_dict[cls, y]:
                n+=1
            pre_tot.append(n/y)
            rap_tot.append(n/11)

        plt.plot(x, pre_tot, marker='o', label='precision')
        plt.plot(x, rap_tot, marker='o', label='rappel')
        plt.title(f"per/rap courbe de classe {cls}")
        plt.legend()
        plt.grid(True)
        plt.show()



print("evaluation")
C = 10 #nombre de classe +1

## KNN
# for mtd in all_vectors.approche:
#     true_labels, pred_labels, proba_labels = knn.knn_leave_one_out(mtd,4)
#     tp, fp, fn = labels_to_tp(true_labels,pred_labels)
#     tab_rap , rapp = rappel(tp,fn)
#     tab_prc,prcc= precision(tp,fp)
#     fs = fscore(rapp,prcc,1)
#     mat = conf_mat(tp, fp, fn)
#     # dessin_mat(mat)
#     # prerap_curve(true_labels,pred_labels,k=10)
#     print(f"rappel {rapp}, precision/taux reconnaissance {prcc}, fscore {fs}")

## Kmeans
#import k_means_v2 as kk
# true_labels, pred_labels, proba_labels = kk.true_labels,kk.pred_labels,kk.proba_labels
# tp, fp, fn = labels_to_tp(true_labels,pred_labels)
# tab_rap , rapp = rappel(tp,fn)
# tab_prc,prcc = precision(tp,fp)
# fs = fscore(rapp,prcc,1)
# print(f"rappel {rapp}, precision {prcc}, fscore {fs}")
# mat=conf_mat(tp,fp,fn)
#dessin_mat(mat)
# tab_rap,rapp = rappel(kk.tp,kk.fn)
# tab_prc,prcc = precision(kk.tp,kk.fp)
# fs = fscore(rapp,prcc,1)
# mat=conf_mat(kk.tp,kk.fp,kk.fn)
# dessin_mat(mat)
# ROC_curve(kk.true_labels, kk.proba_labels)
# prerap_curve(kk.true_labels,kk.pred_labels,k=10)
# print(f"rappel {rapp}, precision {prcc}, fscore {fs}")


## VM
# true_labels, pred_labels, proba_labels = vote_maj.vm_leave_one_out(3)
# tp, fp, fn = labels_to_tp(true_labels,pred_labels)
# tab_rap , rapp = rappel(tp,fn)
# tab_prc,prcc = precision(tp,fp)
# fs = fscore(rapp,prcc,1)
# mat = conf_mat(tp, fp, fn)
# dessin_mat(mat)
# # ROC_curve(true_labels,proba_labels)
# # prerap_curve(true_labels,pred_labels,k=10)
# print(f"rappel {rapp}, precision {prcc}, fscore {fs}")


