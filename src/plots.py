import matplotlib.pyplot as plt
import numpy as np
test=open("data/train-test_compare.txt")
wgan=open("data/train-wgan_compare.txt")
wgan_gp=open("data/train-wgan-gp_compare.txt")

np.array(["1","2"])
# ===== Get the range =====
a1,b1,delta1=np.array(test.readline().split(" ")).astype(np.int)
a2,b2,delta2=np.array(wgan.readline().split(" ")).astype(np.int)
a3,b3,delta3=np.array(wgan_gp.readline().split(" ")).astype(np.int)

# ===== Get data =====
a_test=np.array(test.readline().split(" "))[:-2].astype(np.float)
a_wgan=np.array(wgan.readline().split(" "))[:-2].astype(np.float)
a_wgan_gp=np.array(wgan_gp.readline().split(" "))[:-2].astype(np.float)

# ===== Visualize =====
plt.plot(range(a1,b1,delta1),a_test,'r',label="test")
plt.plot(range(a2,b2,delta2),a_wgan,'b',label="wgan")
plt.plot(range(a3,b3,delta3),a_wgan_gp,'g',label="wgan-gp")
plt.legend()
plt.show()