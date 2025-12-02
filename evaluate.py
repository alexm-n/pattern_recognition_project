import os
from knn import load_vector, knn_classify

def leave_one_out(folderpath, k):
    files = os.listdir(folderpath)
    files.sort()
    database = []
    for filename in files:
        vec = load_vector(os.path.join(folderpath, filename))
        classe = int(filename[1:3])
        database.append((filename, vec, classe))
    predictions = []
    true_labels = []
    pred_labels = []
    confusion = {}
    for test_filename, test_vec, true_classe in database:
        train_db = [(vec, classe) for (fn, vec, classe) in database if fn != test_filename]
        pred_classe = knn_classify(train_db, test_vec, k)
        true_labels.append(true_classe)
        pred_labels.append(pred_classe)
        confusion[(true_classe, pred_classe)] = confusion.get((true_classe, pred_classe), 0) + 1
    accuracy = sum(t == p for t, p in zip(true_labels, pred_labels)) / len(true_labels)
    return accuracy, confusion, true_labels, pred_labels

if __name__ == "__main__":
    folderpath = "./data/F0/"
    k = 5
    a, c, t, p = leave_one_out(folderpath, k)
    print(a)
    print(c)
    print(t)
    print(p)


