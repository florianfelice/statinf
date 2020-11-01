import numpy as np

import os
import sys
import getpass

if sys.platform == 'darwin':
    sys.path.append(f"/Users/{getpass.getuser()}/Documents/statinf/")
elif sys.platform in ['linux', 'linux1', 'linux2']:
    sys.path.append(f"/home/{getpass.getuser()}/statinf/")
else:
    sys.path.append(f"C:/Users/{getpass.getuser()}/Documents/statinf/")

from statinf import stats


a = [30.02, 29.99, 30.11, 29.97, 30.01, 29.99]  # np.random.normal(loc=25, scale=1, size=N)
b = [29.89, 29.93, 29.72, 29.98, 30.02, 29.98]  # np.random.normal(loc=24.8, scale=1, size=N)

tt = stats.ttest(a, mu=30)
print(tt)


tt2 = stats.ttest_2samp(a, b, 0.05, two_sided=True)
print(tt2)


ks1 = stats.kstest(a, b)
print(ks1)

ks2 = stats.kstest(np.random.normal(size=100))
print(ks2)
