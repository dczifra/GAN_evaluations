from keras.models import load_model
from keras.models import model_from_json
import numpy as np
import os

def get_model(model_file):
    # load json and create model
    json_file = open(model_file+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights(model_file+".h5")
    print("Loaded model from disk")
    return loaded_model

def transform_num(elem):
    return str(int(128.0*(1.0+elem[0])))
def generate_samples(model,output_file):
    # ===== Generate samples: =====
    batch_size=1000
    latent_dim=100
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    
    # Generate a batch of new images
    gen_imgs = model.predict(noise)
    print(np.shape(gen_imgs))

    # ===== Write out to file =====
    if(not os.path.isdir(output_file)):
        os.mkdir(output_file)
    iter=0
    for img in gen_imgs:
        myfile=open(output_file+"/image_"+str(iter)+".txt","w")
        for row in img:
            for elem in row:
                myfile.write(transform_num(elem)+" ")
            myfile.write("\n")
        
        iter+=1
        myfile.close()

disc_model=get_model("data/mnist_wgan/discriminator_1000")
#model.summary()

gen_model=get_model("data/mnist_wgan/generator_1000")
#model.summary()
generate_samples(gen_model,"data/mnist/wgan")

gen_model=get_model("data/mnist_wgan-gp/generator_1000")
#model.summary()
generate_samples(gen_model,"data/mnist/wgan-gp")
