import os

def create_data_for_plot():
    os.system("./bin/main --ini src/params/params_wgan_1000.ini")
    os.system("./bin/main --ini src/params/params_wgan_5000.ini")
    os.system("./bin/main --ini src/params/params_wgan_10000.ini")

    os.system("./bin/main --ini src/params/params_wgan-gp_1000.ini")
    os.system("./bin/main --ini src/params/params_wgan-gp_5000.ini")
    os.system("./bin/main --ini src/params/params_wgan-gp_10000.ini")

def do_plot():
    1

create_data_for_plot()