from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve,auc,precision_recall_curve

import knn
import vote_maj
import k_means_v2 as kk


def labels_to_tp(true_labels, pred_labels):
    tp = {c: 0 for c in range(1, 10)}
    fp = {(p, t): 0 for p in range(1, 10) for t in range(1, 10)}
    fn = {(t, p): 0 for t in range(1, 10) for p in range(1, 10)}
    for x in range(1,10):
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
    matrice = [[0 for i in range(9)] for i in range(9)]
    for i in range(1,10):
        matrice[i-1][i-1] = tp[i]
        for x in range(i+1,10):
            matrice[x-1][i-1]=fp[x,i]
            matrice[i-1][x-1]=fn[i,x]
    return matrice

def dessin_mat(mat):
    plt.imshow(mat, cmap="Blues")
    plt.colorbar()

    # Ajouter les valeurs dans les cases
    for x in range(9):
        for y in range(len(mat[0])):
            plt.text(x, y, mat[x][y], ha='center', va='center', color='black')

    plt.xlabel("Estimation")
    plt.ylabel("Référence")
    plt.title("Matrice de confusion")
    plt.show()

def ROC_curve(true_labels_dict, proba_labels_dict):
    true_labels = []
    proba_labels = []
    for x in range(1,10):
        for y in range(1,12):
            true_labels.append(true_labels_dict[x, y])
            proba_labels.append(proba_labels_dict[x, y])
    for cls in range(1, 10):
        binaire_true = [1 if x == cls else 0 for x in true_labels]
        binaire_prob = [prb[cls - 1] for prb in proba_labels]

        fpr, tpr, _ = roc_curve(binaire_true, binaire_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve for class {cls}")
        plt.show()

def prc_rap_curve(true_labels_dict, proba_labels_dict):
    true_labels = []
    proba_labels = []
    for x in range(1,10):
        for y in range(1,12):
            true_labels.append(true_labels_dict[x,y])
            proba_labels.append(proba_labels_dict[x,y])
    for cls in range(1,10):
        binaire_true = [1 if x == cls else 0 for x in true_labels]
        binaire_prob = [prb[cls - 1] for prb in proba_labels]

        prcs, recall, _ = precision_recall_curve(binaire_true, binaire_prob)
        plt.plot(recall, prcs)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision–Recall Curve for class {cls}")
        plt.show()


print("evaluation")

#
# true_labels, pred_labels, proba_labels = knn.knn_leave_one_out('E34',4)
# tp, fp, fn = labels_to_tp(true_labels,pred_labels)
# tab_rap , rapp = rappel(tp,fn)
# tab_prc,prcc= precision(tp,fp)
# fs = fscore(rapp,prcc,1)
# mat = conf_mat(tp, fp, fn)
# dessin_mat(mat)
# ROC_curve(true_labels,proba_labels)
# prc_rap_curve(true_labels,proba_labels)
# print(f"rappel {rapp}, precision {prcc}, fscore {fs}")


# true_labels, pred_labels, proba_labels = kk.true_labels,kk.pred_labels,kk.proba_labels
# tp, fp, fn = labels_to_tp(true_labels,pred_labels)
# tab_rap , rapp = rappel(tp,fn)
# tab_prc,prcc = precision(tp,fp)
# fs = fscore(rapp,prcc,1)
# print(f"rappel {rapp}, precision {prcc}, fscore {fs}")
# mat=conf_mat(tp,fp,fn)
# #dessin_mat(mat)
# tab_rap,rapp = rappel(kk.tp,kk.fn)
# tab_prc,prcc = precision(kk.tp,kk.fp)
# fs = fscore(rapp,prcc,1)
# tp, fp, fn = labels_to_tp(true_labels,pred_labels)
# mat=conf_mat(kk.tp,kk.fp,kk.fn)
# #dessin_mat(mat)
# ROC_curve(true_labels,proba_labels)
# prc_rap_curve(true_labels,proba_labels)
#print(f"rappel {rapp}, precision {prcc}, fscore {fs}")

#
# true_labels, pred_labels, proba_labels = vote_maj.vm_leave_one_out(4)
# tp, fp, fn = labels_to_tp(true_labels,pred_labels)
# tab_rap , rapp = rappel(tp,fn)
# tab_prc,prcc = precision(tp,fp)
# fs = fscore(rapp,prcc,1)
# mat = conf_mat(tp, fp, fn)
# dessin_mat(mat)
# ROC_curve(true_labels,proba_labels)
# prc_rap_curve(true_labels,proba_labels)
#print(f"rappel {rapp}, precision {prcc}, fscore {fs}")

