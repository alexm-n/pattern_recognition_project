from matplotlib import pyplot as plt
from k_means_v2 import tp,fp,fn


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


# matrice = conf_mat(tp,fn)
# dessin_mat(matrice)
# print(matrice)