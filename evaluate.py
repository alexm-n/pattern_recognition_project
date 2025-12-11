# import os
# import copy
# from knn import knn_classify
# import all_vectors as alldata
# 
# def knn_leave_one_out(method, k):
#     true_labels = []
#     pred_labels = []
#     for (test_classe, test_ech) in alldata.approche[method]:
#         test_vec = alldata.approche[method][(test_classe, test_ech)]
#         train_db = copy.deepcopy(alldata.approche[method])
#         train_db.pop((test_classe, test_ech))
#         pred_classe = knn_classify(train_db, test_vec, k)
#         true_labels.append(test_classe)
#         pred_labels.append(pred_classe)
#     return true_labels, pred_labels
# 
# def metriques(true_labels, pred_labels):
#     tp = {}
#     fp = {}
#     fn = {}
#     for (true_classe, pred_classe) in zip(true_labels, pred_labels):
#         tp[true_classe] = tp.get(true_classe, 0) + (true_classe == pred_classe)
#         fp[true_classe] = fp.get(true_classe, 0) + (true_classe != pred_classe)
#         fn[pred_classe] = fn.get(pred_classe, 0) + (true_classe != pred_classe)
#     return tp, fp, fn
# 
# 
# if __name__ == "__main__":
#     true_labels, pred_labels = knn_leave_one_out("E34", 5)
#     tp, fp, fn = metriques(true_labels, pred_labels)
#     print("tp", tp)
#     print("fp", fp)
#     print("fn", fn)    
# 
