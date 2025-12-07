import os

from matplotlib import pyplot as plt
#from k_means_v2 import tp,fp,fn
import evaluate
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

from knn import load_vector, euclidean_dist

def labels_to_tp(true_labels, pred_labels):
    tp = {c: 0 for c in range(1, 10)}
    fp = {(p, t): 0 for p in range(1, 10) for t in range(1, 10)}
    fn = {(t, p): 0 for t in range(1, 10) for p in range(1, 10)}

    for x, y in zip(true_labels, pred_labels):
        if x == y:
            tp[x] += 1
        else:
            fn[(x, y)] += 1
            fp[(y, x)] += 1

    return tp, fp, fn
def tp_to_labels(tp,fp):
    true_labels, pred_labels = [],[]
    for x, nbr in tp.items():
        true_labels.extend([x] * nbr)
        pred_labels.extend([x] * nbr)
    for (x, y), nbr in fp.items():
        true_labels.extend([y] * nbr)
        pred_labels.extend([x] * nbr)
    return true_labels, pred_labels

def rappel(tp, fn):
    r = {}
    r_glob = 0
    for k in tp:
        r[k] = tp[k] / (tp[k] + fn[k])
        r_glob += r[k]
    r_glob = r_glob / len(tp)
    return r, r_glob


def precision(tp, fp):
    r = {}
    r_glob = 0
    for k in tp:
        r[k] = tp[k] / (tp[k] + fp[k])
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

def ROC_curve(true_labels, pred_labels, proba_labels):
    for cls in range(1, 10):
        binaire_true = [1 if x == cls else 0 for x in true_labels]
        binaire_prob = [prb if true_cls == cls else 1 - prb for prb, true_cls in zip(proba_labels, true_labels)]

        fpr, tpr, _ = roc_curve(binaire_true, binaire_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve for class {cls}")
        plt.show()

def prc_rap_curve(true_labels, pred_labels, proba_labels):
    for cls in range(1,10):
        binaire_true = [1 if x == cls else 0 for x in true_labels]
        binaire_prob = [prb if true_cls == cls else 1 - prb for prb, true_cls in zip(proba_labels, true_labels)]

        prcs, recall, _ = precision_recall_curve(binaire_true, binaire_prob)
        plt.plot(recall, prcs)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision–Recall Curve for class {cls}")
        plt.show()


def knn_classify(database, query_vec, k):
  distances = [(euclidean_dist(query_vec, data_vec), nclasse) for (data_vec, nclasse) in database]
  distances.sort(key=lambda x: x[0])
  k_nearest = distances[:k]
  count = {}
  for _, nclasse in k_nearest:
    count[nclasse] = count.get(nclasse, 0) + 1
  return count
def leave_one_out(folderpath, k):
    files = os.listdir(folderpath)
    files.sort()
    database = []
    for filename in files:
        vec = load_vector(os.path.join(folderpath, filename))
        classe = int(filename[1:3])
        database.append((filename, vec, classe))
    prob_labels = []
    true_labels = []
    pred_labels = []
    confusion = {}
    for test_filename, test_vec, true_classe in database:
        train_db = [(vec, classe) for (fn, vec, classe) in database if fn != test_filename]
        count = knn_classify(train_db, test_vec, k)
        pred_classe = max(count, key=count.get)
        vr =0
        vr = count.get(true_classe, 0)
        proba_classe = vr/k
        true_labels.append(true_classe)
        pred_labels.append(pred_classe)
        prob_labels.append(proba_classe)
        confusion[(true_classe, pred_classe)] = confusion.get((true_classe, pred_classe), 0) + 1
    accuracy = sum(t == p for t, p in zip(true_labels, pred_labels)) / len(true_labels)
    return accuracy, confusion, true_labels, pred_labels , prob_labels
folderpath = "C:/Users/user/OneDrive/Bureau/AnUniv/m1-p/reconnaissance formes/proj/F0/"
k = 5
a, c, t, p , prb= leave_one_out(folderpath, k)
print(a)
print(c)
print(t)
print(p)
tp,fp,fn = labels_to_tp(t,p)
x,y = tp_to_labels(tp,fp)
print(x)
print(y)





