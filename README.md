# GAN evaluation metrics

This is the README of GAN_evaluations git repo, and containes the code base for my *Master Thesis* :
https://web.cs.elte.hu/blobs/diplomamunkak/msc_alkmat/2019/czifra_domonkos.pdf

## Compile C++ codes:

```
g++ -O3 -std=c++17 -o bin/main src/main.cpp -lstdc++fs
```

* This command compiles the **src** folder, included the *Hungarian method*, *Mincost matching*, *Mincost transportation problem*, and different evaluation metrics. The binary compiles into **bin** folder.

## Evaluation script:

Run a specific modell with:

```
python scripts/process_model.py -N 1000 -r 500 -choose toy -batchs celeba_49999 -dataset celeba -size 64 64 3
```

Parameters:
* **-N**: Number of samples in the metrics
* **-r**: Range, runs until **N**
* **-choose**: Mode type
* **-batch**: Model name
* **-dataset**: Chosen dataset
* **-size**: Size of images, divided by spaces (For example: 64 64 3)

* The plots are generated automatically, but you can invoke it manually:

## Plots:
    python scripts/plot.py -N 1000 -r 500 -choose toy -batchs celeba_49999 -dataset celeba -size 64 64 3

## Synchronization:
 There is a synchronization script, if you don't have *-X* in terminal.
    
    python scripts/syncronization.py
    