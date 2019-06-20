import numpy as np
import matplotlib.pyplot as plt
# plot coverage
num = range(50)

# plot acc
acc = 0.7953
index = np.array([5 * i for i in range(5, 36)])
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
dr = dr[:,5:]
ds = ds[:,5:]
# dr_mean = np.mean(dr, axis=0)
# ds_mean = np.mean(ds, axis=0)
dr_mean = np.sqrt(np.mean(np.square(dr), axis=0))
ds_mean = np.sqrt(np.mean(np.square(ds), axis=0))
plt.plot(index, dr_mean, c='k')
plt.plot(index, ds_mean, c='r')
plt.show()

random = np.array(random)
select = np.array(select)
random = random[:,5:]
select = select[:,5:]
r_mean = np.mean(np.array(random), axis=0)
s_mean = np.mean(np.array(select), axis=0)
plt.plot(index, r_mean, c='k')
plt.plot(index, s_mean, c='r')
plt.show()