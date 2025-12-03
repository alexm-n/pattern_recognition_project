import numpy as np
import all_vectors as vec
import random
import copy

n_patterns = 99


def euclidian_dist(x, y):
    return np.sum(np.square(np.subtract(x, y)))


for classe in vec.approche:
    cluster = {}
    means = {}

    print(f"\n------------- Pour {classe} -----------------")
    for k in range(9, 13):
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

        tp, fn, fp = {}, {}, {}
        for x in range(1, 10):
            tp[x] = 0
            for y in range(1, 10):
                fp[x, y] = 0
                fn[x, y] = 0
        # final results for each k
        print(f"\noptimum value pour k={parted} iteration {num_itr} avec minimum squarred error :{local_min}")
        for i in cluster:
            print(f"Pour cluster {i} il a {len(cluster[i])} points sont {cluster[i].keys()}")
            print(f"le centroid est {means[i]}")

            # partie test
            for x, y in cluster[i]:
                if x == i:
                    tp[i] += 1
                else:
                    fp[i, x] += 1
                    fn[x, i] += 1

        print("tp", tp)
        print("fp", fp)
        print("fn", fn)



        break
