import pickle
f = open('/home/ljy/continue-completing-cycle/data/FB15K/active_learning/init_triples.pkl', 'rb')
a = pickle.load(f)

print(a[0])