import matplotlib.pyplot as plt
import numpy as np

def different_model_compare( file_test,
            file_wgan,
            file_wgan_gp,
            title,
            labels):
    test=open(file_test)
    wgan=open(file_wgan)
    wgan_gp=open(file_wgan_gp)

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
    plt.plot(range(a1,b1,delta1),a_test,'r',label=labels[0])
    plt.plot(range(a2,b2,delta2),a_wgan,'b',label=labels[1])
    plt.plot(range(a3,b3,delta3),a_wgan_gp,'g',label=labels[2])
    
    plt.xlabel("Minta meret")
    plt.ylabel("[Párosítás]/[Minta meret]")
    plt.title(title)
    plt.legend()
    plt.show()

title="WGAN, WGAN-GP Párosítás-Pontszám 1000 EPOCH"
labels=["test","wgan","wgan-gp"]
different_model_compare( file_test="data/plot/train-test_compare.txt",
        file_wgan="data/plot/train-wgan_compare_1000.txt",
        file_wgan_gp="data/plot/train-wgan-gp_compare_1000.txt",
        title=title,
        labels=labels)

title="WGAN, WGAN-GP Párosítás-Pontszám 5000 EPOCH"
different_model_compare( file_test="data/plot/train-test_compare.txt",
        file_wgan="data/plot/train-wgan_compare_5000.txt",
        file_wgan_gp="data/plot/train-wgan-gp_compare_5000.txt",
        title=title,
        labels=labels)

title="WGAN, WGAN-GP Párosítás-Pontszám 10000 EPOCH"
different_model_compare( file_test="data/plot/train-test_compare.txt",
        file_wgan="data/plot/train-wgan_compare_10000.txt",
        file_wgan_gp="data/plot/train-wgan-gp_compare_10000.txt",
        title=title,
        labels=labels)

title="WGAN Párosítás-Pontszám "
labels=["wgan 1000","wgan 5000","wgan 10000"]
different_model_compare( file_test="data/plot/train-wgan_compare_1000.txt",
        file_wgan="data/plot/train-wgan_compare_5000.txt",
        file_wgan_gp="data/plot/train-wgan_compare_10000.txt",
        title=title,
        labels=labels)

title="WGAN, WGAN-GP Deficit-Pontszám"
labels=["test","wgan","wgan-gp"]
different_model_compare( file_test="data/plot/train-wgan-gp_compare_10000.txtdeficit",
        file_wgan="data/plot/train-wgan-gp_compare_5000.txtdeficit",
        file_wgan_gp="data/plot/train-wgan-gp_compare_1000.txtdeficit",
        title=title,
        labels=labels)