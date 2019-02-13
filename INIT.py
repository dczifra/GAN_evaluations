# ===== Get 3PP products: =====
import subprocess
import os
import tensorflow as tf
if(not os.path.isdir("munkres-tensorflow")):
    subprocess.call(["git", "clone","https://github.com/mbaradad/munkres-tensorflow.git"])
    os.chdir("./munkres-tensorflow")
    TF_INC=tf.sysconfig.get_include()
    #g++ -std=c++11 -shared hungarian.cc -o hungarian.so -fPIC -I$TF_INC -L$TF_LIB -ltensorflow_framework -D_GLIBCXX_USE_CXX11_ABI=0
    os.system("g++ -std=c++11 -shared hungarian.cc -o hungarian.so -fPIC -I "+TF_INC)
    os.chdir("..")

exit(1)

sample_size=1000
# ===== Generate mnist train/test datasets =====
from scripts.post_model import generate_mnist,generate_samples,get_model
import os
generate_mnist(sample_size,True)
generate_mnist(sample_size,False)

# ===== Generate data from GAN models =====
gen_model=get_model("models/wgan_1000/generator_1000")
generate_samples(gen_model,"data/mnist/wgan",sample_size)

gen_model=get_model("models/wgan-gp_1000/generator_1000")
generate_samples(gen_model,"data/mnist/wgan-gp",sample_size)

