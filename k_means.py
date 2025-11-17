import numpy as np
import all_vectors as vec
import random

n_patterns = 99

def euclidian_dist(x,y):
    return np.sum(np.square(np.subtract(x,y)))

for classe in vec.approche :
    cluster={}
    means={}
    print(f"\n------------- Pour {classe} -----------------")
    for k in range(2,13):
        local_min, parted ,num_itr,stop= 0, 0,0, False

        #initiale partition cluster
        nbr_dans_partition = n_patterns//k
        i=0
        cls=1
        temp = vec.approche[classe].copy()
        for cls in range(1,k+1):
            cluster[cls] = {}
            if cls==k:
                for x,y in temp:
                    cluster[cls][x, y] = temp[x, y]
            else:
                for i in range(nbr_dans_partition):
                    x,y=0,0
                    while (x,y) not in temp:
                        x = random.randint(1, 9)
                        y = random.randint(1, 11)
                    cluster[cls][x, y] = temp[x, y]
                    temp.pop((x,y))

        itr=0
        chance=0

        while(not stop):
            #calcul means
            for i in range(1,k+1):
                n = len(cluster[i])
                tot = [0] * len(vec.approche[classe][1, 1])
                for x,y in cluster[i]:
                    tot = np.add(tot,cluster[i][x,y])

                if n == 0: n=1
                means[i] = np.divide(tot,n).tolist()


            #calcul mean erro
            squared_error = 0
            for i in range(1,k+1):
                for x,y in cluster[i]:
                    squared_error += np.sqrt(euclidian_dist(cluster[i][x,y],means[i]))

            if local_min>squared_error or itr==0:
                local_min,parted,num_itr = (squared_error,k,itr)
            else:
                    chance += 1
            itr += 1
            if itr > 10 or chance == 3:
                stop = True
                break

            # nv partition
            for i in range(1, k + 1):
                cluster[i]={}
            for (x,y) in vec.approche[classe]:
                distance=[]
                for i in range(1, k + 1):
                    distance.append(euclidian_dist(vec.approche[classe][x, y],means[i]))
                cls = distance.index(min(distance))+1
                cluster[cls][x, y] = vec.approche[classe][x, y]


        print(f"\noptimum value pour k={parted} iteration {num_itr} avec squarred error :{local_min}")
        for i in cluster:
            print(f"Pour cluster {i} il a {len(cluster[i])} points sont {cluster[i].keys()}")
            print(f"le centroid est {means[i]}")




