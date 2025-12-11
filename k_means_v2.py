import numpy as np
import all_vectors as vec
import random
import copy


def init_cluster(method, k, nbr):
    #la premiere initialisation des clusters
    cluster = {}
    temp = copy.deepcopy(vec.approche[method])
    for cls in range(1, k + 1):
        cluster[cls] = {}
        if cls == k: #puisque c'est le dernier cluster on mets tout les points restant
            cluster[cls] = copy.deepcopy(temp)
        else:
            for i in range(nbr):
                x, y = 0, 0
                while (x, y) not in temp: # pour chaque cluster on y mets nbr points de facon aleotaoire et se trouvant dans temp
                    x = random.randint(1, 9)
                    y = random.randint(1, 11)
                cluster[cls][x, y] = temp[x, y]
                temp.pop((x, y))
    return cluster


def get_mean(cluster, k, method):
    #trouver le centroide de chaque cluster
    means = {}
    for i in range(1, k + 1):
        n = len(cluster[i])
        tot = [0] * len(vec.approche[method][1, 1])
        for x, y in cluster[i]:
            tot = np.add(tot, cluster[i][x, y]) #additionnant tout les point du cluster

        if n == 0: n = 1
        means[i] = np.divide(tot, n).tolist() #on divise par le nombre de points dans le cluster
    return means


def euclidian_dist(x, y):
    #la distance euclidienne au carre
    return np.sum(np.square(np.subtract(x, y)))


def nv_cluster(k, method, means):
    #a la fin de la boucle si condition non rempie on cherche les nouveaux points des clusters
    cluster = {}
    for i in range(1, k + 1):
        cluster[i] = {}
    for (x, y) in vec.approche[method]:
        distance = []
        for i in range(1, k + 1):
            distance.append(euclidian_dist(vec.approche[method][x, y], means[i])) # pour chaque point en calcule la distance au centroide de chaque cluster
        cls = distance.index(min(distance)) + 1 #on prend le minimum et on y ajoute le point
        cluster[cls][x, y] = vec.approche[method][x, y]
    return cluster


def class_des_cluster(cluster): #chaque cluster a une classe qu'il represente, c"est la classe majoritaire
    clus_clas = {}
    for i in cluster:
        if len(cluster[i]) == 0:#si aucun point on lui donne une classe au hasard
            clus_clas[i] = random.randint(1, 9)
        else:
            counts = {}
            for (x, y) in cluster[i]:
                counts[x] = counts.get(x, 0) + 1
            clus_clas[i] = max(counts, key=counts.get)# on prend la classe avec le plus de points de cette classe
    return clus_clas


def squared_err(cluster, k, means):
    squared_error = 0
    for i in range(1, k + 1):
        for x, y in cluster[i]:
            squared_error += euclidian_dist(cluster[i][x, y], means[i]) #la somme des distance de chaque point du cluster a son centroide
    return squared_error


def get_labels(cluster, clus_clas):#trouver les labels
    true_labels = {}
    pred_labels = {}
    for x in range(1, 10):
        for y in range(1, 12):
            true_labels[x, y] = x #pour chaque on connait deja sa vrai classe
            for i in cluster:
                if (x, y) in cluster[i]:
                    pred_labels[x, y] = clus_clas[i] #on lui donne au point la classe representé par le cluster ou elle se trouve
    return true_labels, pred_labels


def kmeans_classify(method, k):
    print(f"\n------------- Pour {method} -----------------")
    means = {}
    local_min, parted, num_itr, stop = 0, 0, 0, False
    n_patterns = 99

    # initiale partition cluster
    nbr_dans_partition = n_patterns // k #trouver le nombre moyen de points dans chaque cluster
    cluster = init_cluster(method, k, nbr_dans_partition)

    itr, chance = 0, 0

    while not stop:
        # calcul means
        means = get_mean(cluster, k, method)

        # calcul mean error
        squared_error = squared_err(cluster, k, means)

        if local_min > squared_error or itr == 0:#verifier que l'erreur a diminuer
            local_min, parted, num_itr = (squared_error, k, itr)
            chance=0
        else:
            chance += 1
        itr += 1
        if itr > 10 or chance == 3:# si durant 3 fois il ne diminue pas l'erreur ou on depasse 10 iterations on arrete la boucle 
            stop = True
            break

        # nv partition
        cluster = nv_cluster(k, method, means)

    #trouver la classes des clusters
    clus_clas = class_des_cluster(cluster)

    # resumé des resultats
    print(f"\noptimum value pour k={parted} iteration {num_itr} avec minimum squarred error :{local_min}")
    for i in cluster:
        print(
            f"Pour cluster {i} il a {len(cluster[i])}, avec comme classe majoritaire {clus_clas[i]} et les points sont {cluster[i].keys()}")
        print(f"le centroid est {means[i]}")

    # calcul des labels
    true_labels, pred_labels = get_labels(cluster, clus_clas)

    return true_labels, pred_labels
