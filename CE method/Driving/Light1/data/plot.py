import numpy as np
import matplotlib.pyplot as plt
# plot coverage
num = range(50)
random_cov = []
select_cov = []
index = np.array([5 * i for i in range(1, 31)])
for i in num:
    cov1 = np.loadtxt("random_cov{}.csv".format(i))
    cov2 = np.loadtxt("select_cov{}.csv".format(i))
    random_cov.append(cov1)
    select_cov.append(cov2)
random = np.array(random_cov)
select = np.array(select_cov)
cov1 = np.mean(random, axis=0)
cov2 = np.mean(select, axis=0)
plt.subplot(121)
plt.plot(index, cov1, c='k')
plt.plot(index, cov2, c='r')


# plot acc
acc = 0.15315013043537815
index = np.array([5 * i for i in range(1, 31)])
ds = []
dr = []
random = []
select = []
# num = 30
for i in num:
    r = np.loadtxt("random{}.csv".format(i))
    s = np.loadtxt("select{}.csv".format(i))
    random.append(r)
    select.append(s)
    dr.append(np.abs(r - acc))
    ds.append(np.abs(s - acc))

dr = np.array(dr)
ds = np.array(ds)
# dr_mean = np.mean(dr, axis=0)
# ds_mean = np.mean(ds, axis=0)
dr_mean = np.sqrt(np.mean(np.square(dr), axis=0))
ds_mean = np.sqrt(np.mean(np.square(ds), axis=0))
plt.subplot(122)
plt.plot(index, dr_mean, c='k')
plt.plot(index, ds_mean, c='r')
plt.show()

print(np.mean(np.square(ds_mean/dr_mean)))