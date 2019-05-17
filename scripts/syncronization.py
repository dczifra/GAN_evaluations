import os
def copy_pngs(folder, dest):
    new_dest = os.path.join(dest,folder)
    for file in os.listdir(folder):
        #print(file, new_dest)
        if('.png' in str(file)):
            os.system("mkdir -p {}".format(new_dest))
            os.system("cp {} {}".format(os.path.join(folder,file), os.path.join(new_dest, file) ) )
            #os.system("ssh -p 2820 doma945@renyi.hu \"mkdir -p ./www/temp/{}\" ".format(new_dest))
            #print("ssh -p 2820 doma945@renyi.hu \"mkdir -r ./www/temp/{}\" ".format(new_dest))
            #os.system("scp -r -P 2820 {} renyi.hu:./www/temp/{}".format(
            #        os.path.join(folder,file),
            #        os.path.join(new_dest, file)))
        elif(os.path.isdir(os.path.join(folder, file))):
            #print(os.path.join(folder, file))
            copy_pngs(os.path.join(folder, file), dest)

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Syncronization between www/temp and geforce')
    parser.add_argument('-from', metavar='STRING', dest="src",
                            type = str, help='the required fictures from', default="models")
    return parser.parse_args()

args = get_args()
source = args.src
to = source.split("/")[-1]

import subprocess
myfile = open(".trash.txt", "w")
os.system("mkdir {}_temp".format(to))
copy_pngs(source, "{}_temp".format(to))
subprocess.run("scp -r -P 2820 {}_temp login.renyi.hu:./www/temp".format(to).split(' '), stdout=myfile)
os.system("ls {}_temp".format(to))
os.system("rm -r {}_temp".format(to))
