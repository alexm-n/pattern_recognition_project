import numpy as np

import all_vectors
import all_vectors as vec
import random
import copy

n_patterns = 99
k=9

def euclidian_dist(x, y):
    return np.sum(np.square(np.subtract(x, y)))


for classe in ['E34']:#vec.approche:


    print(f"\n------------- Pour {classe} -----------------")
    for k in range(9, 13):
        cluster = {}
        means = {}
        local_min, parted, num_itr, stop = 0, 0, 0, False

        # initiale partition cluster
        nbr_dans_partition = n_patterns // k
        cls = 1
        temp = copy.deepcopy(vec.approche[classe])
        for cls in range(1, k + 1):
            cluster[cls] = {}
            if cls == k:
                for x, y in temp:
                    cluster[cls] = copy.deepcopy(temp)
            else:
                for i in range(nbr_dans_partition):
                    x, y = 0, 0
                    while (x, y) not in temp:
                        x = random.randint(1, 9)
                        y = random.randint(1, 11)
                    cluster[cls][x, y] = temp[x, y]
                    temp.pop((x, y))

        itr = 0
        chance = 0

        while (not stop):
            # calcul means
            for i in range(1, k + 1):
                n = len(cluster[i])
                tot = [0] * len(vec.approche[classe][1, 1])
                for x, y in cluster[i]:
                    tot = np.add(tot, cluster[i][x, y])

                if n == 0: n = 1
                means[i] = np.divide(tot, n).tolist()

            # calcul mean erro
            squared_error = 0
            for i in range(1, k + 1):
                for x, y in cluster[i]:
                    squared_error += euclidian_dist(cluster[i][x, y], means[i])

            if local_min > squared_error or itr == 0:
                local_min, parted, num_itr = (squared_error, k, itr)
            else:
                chance += 1
            itr += 1
            if itr > 10 or chance == 3:
                stop = True
                break

            # nv partition
            for i in range(1, k + 1):
                cluster[i] = {}
            for (x, y) in vec.approche[classe]:
                distance = []
                for i in range(1, k + 1):
                    distance.append(euclidian_dist(vec.approche[classe][x, y], means[i]))
                cls = distance.index(min(distance)) + 1
                cluster[cls][x, y] = vec.approche[classe][x, y]

        tp = {c: 0 for c in range(1, 10)}
        fp = {(p, t): 0 for p in range(1, 10) for t in range(1, 10)}
        fn = {(t, p): 0 for t in range(1, 10) for p in range(1, 10)}

        clus_clas = {}
        for i in cluster:
            if len(cluster[i]) == 0:
                clus_clas[i] = random.randint(1, 9)
            else:
                counts = {}
                for (x, y) in cluster[i]:
                    counts[x] = counts.get(x, 0) + 1
                clus_clas[i] = max(counts, key=counts.get)
        print(clus_clas)

        # final results for each k
        print(f"\noptimum value pour k={parted} iteration {num_itr} avec minimum squarred error :{local_min}")
        for i in cluster:
            print(f"Pour cluster {i} il a {len(cluster[i])}, avec comme classe majoritaire {clus_clas[i]} et les points sont {cluster[i].keys()}")
            print(f"le centroid est {means[i]}")

            # partie test


            for x, y in cluster[i]:
                if x == clus_clas[i]:
                    tp[x] += 1
                else:
                    fp[clus_clas[i], x] += 1
                    fn[x, clus_clas[i]] += 1

        print("tp", tp)
        print("fp", fp)
        print("fn", fn)

        #les labels
        true_labels = {}
        pred_labels = {}
        proba_labels = {}
        for x in range(1,10):
            for y in range(1,12):
                true_labels[x,y] = x
                prob=[]
                norm_proba=[]
                prob_class={}
                for i in cluster:
                    d=euclidian_dist(means[i],all_vectors.approche[classe][x,y])
                    if d!=0 : prob.append(1/d)
                    else : prob.append(1)

                    if (x,y) in cluster[i]:
                        pred_labels[x,y] = clus_clas[i]
                tot = sum(prob)
                for p in prob:
                    norm_proba.append(p/tot)
                for i in cluster:
                    prob_class[clus_clas[i]] = prob_class.get(clus_clas[i],0) + norm_proba[i-1]

                proba_labels[x, y] = [prob_class.get(c, 0) for c in range(1, 10)]

        print(proba_labels)
        break
