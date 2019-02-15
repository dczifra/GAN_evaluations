
sample_size=2000
# ===== Generate mnist train/test datasets =====
from scripts.post_model import generate_mnist,generate_samples,get_model
import os
generate_mnist(sample_size,True)
generate_mnist(sample_size,False)

# ===== Generate data from GAN models 1000 =====
gen_model=get_model("models/wgan/generator_1000")
generate_samples(gen_model,"data/mnist/wgan_1000",sample_size)

gen_model=get_model("models/wgan-gp/generator_1000")
generate_samples(gen_model,"data/mnist/wgan-gp_1000",sample_size)

# ===== Generate data from GAN models 5000 =====
gen_model=get_model("models/wgan/generator_5000")
generate_samples(gen_model,"data/mnist/wgan_5000",sample_size)

gen_model=get_model("models/wgan-gp/generator_5000")
generate_samples(gen_model,"data/mnist/wgan-gp_5000",sample_size)

# ===== Generate data from GAN models 10000 =====
gen_model=get_model("models/wgan/generator_10000")
generate_samples(gen_model,"data/mnist/wgan_10000",sample_size)

gen_model=get_model("models/wgan-gp/generator_10000")
generate_samples(gen_model,"data/mnist/wgan-gp_10000",sample_size)
