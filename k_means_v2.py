import numpy as np
import all_vectors as vec
import random
import copy


def init_cluster(classe, k, nbr):
    cluster = {}
    temp = copy.deepcopy(vec.approche[classe])
    for cls in range(1, k + 1):
        cluster[cls] = {}
        if cls == k:
            cluster[cls] = copy.deepcopy(temp)
        else:
            for i in range(nbr):
                x, y = 0, 0
                while (x, y) not in temp:
                    x = random.randint(1, 9)
                    y = random.randint(1, 11)
                cluster[cls][x, y] = temp[x, y]
                temp.pop((x, y))
    return cluster


def get_mean(cluster, k, classe):
    means = {}
    for i in range(1, k + 1):
        n = len(cluster[i])
        tot = [0] * len(vec.approche[classe][1, 1])
        for x, y in cluster[i]:
            tot = np.add(tot, cluster[i][x, y])

        if n == 0: n = 1
        means[i] = np.divide(tot, n).tolist()
    return means


def euclidian_dist(x, y):
    return np.sum(np.square(np.subtract(x, y)))


def nv_cluster(k, classe, means):
    cluster = {}
    for i in range(1, k + 1):
        cluster[i] = {}
    for (x, y) in vec.approche[classe]:
        distance = []
        for i in range(1, k + 1):
            distance.append(euclidian_dist(vec.approche[classe][x, y], means[i]))
        cls = distance.index(min(distance)) + 1
        cluster[cls][x, y] = vec.approche[classe][x, y]
    return cluster


def class_des_cluster(cluster):
    clus_clas = {}
    for i in cluster:
        if len(cluster[i]) == 0:
            clus_clas[i] = random.randint(1, 9)
        else:
            counts = {}
            for (x, y) in cluster[i]:
                counts[x] = counts.get(x, 0) + 1
            clus_clas[i] = max(counts, key=counts.get)
    return clus_clas


def squared_err(cluster, k, means):
    squared_error = 0
    for i in range(1, k + 1):
        for x, y in cluster[i]:
            squared_error += euclidian_dist(cluster[i][x, y], means[i])
    return squared_error


def get_labels(cluster, clus_clas):
    true_labels = {}
    pred_labels = {}
    for x in range(1, 10):
        for y in range(1, 12):
            true_labels[x, y] = x
            for i in cluster:
                if (x, y) in cluster[i]:
                    pred_labels[x, y] = clus_clas[i]
    return true_labels, pred_labels


def kmeans_classify(classe, k):
    print(f"\n------------- Pour {classe} -----------------")
    means = {}
    local_min, parted, num_itr, stop = 0, 0, 0, False
    n_patterns = 99

    # initiale partition cluster
    nbr_dans_partition = n_patterns // k
    cluster = init_cluster(classe, k, nbr_dans_partition)

    itr, chance = 0, 0

    while not stop:
        # calcul means
        means = get_mean(cluster, k, classe)

        # calcul mean erro
        squared_error = squared_err(cluster, k, means)

        if local_min > squared_error or itr == 0:
            local_min, parted, num_itr = (squared_error, k, itr)
        else:
            chance += 1
        itr += 1
        if itr > 10 or chance == 3:
            stop = True
            break

        # nv partition
        cluster = nv_cluster(k, classe, means)

    clus_clas = class_des_cluster(cluster)

    # final results for each k
    print(f"\noptimum value pour k={parted} iteration {num_itr} avec minimum squarred error :{local_min}")
    for i in cluster:
        print(
            f"Pour cluster {i} il a {len(cluster[i])}, avec comme classe majoritaire {clus_clas[i]} et les points sont {cluster[i].keys()}")
        print(f"le centroid est {means[i]}")

    # les labels
    true_labels, pred_labels = get_labels(cluster, clus_clas)

    return true_labels, pred_labels
