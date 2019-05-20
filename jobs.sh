CUDA_VISIBLE_DEVICE=1 python scripts/process_model.py -N 1000 -r 100 -choose toy -batchs generator_1000 -dataset mnist -size 28 28 1 >out.cout
CUDA_VISIBLE_DEVICE=1 python scripts/process_model.py -N 1000 -r 100 -choose toy -batchs generator_2000 -dataset mnist -size 28 28 1 >out.cout
CUDA_VISIBLE_DEVICE=1 python scripts/process_model.py -N 1000 -r 100 -choose toy -batchs generator_3000 -dataset mnist -size 28 28 1 >out.cout
CUDA_VISIBLE_DEVICE=1 python scripts/process_model.py -N 1000 -r 100 -choose toy -batchs generator_5000 -dataset mnist -size 28 28 1 >out.cout
CUDA_VISIBLE_DEVICE=1 python scripts/process_model.py -N 1000 -r 100 -choose toy -batchs generator_7000 -dataset mnist -size 28 28 1 >out.cout
CUDA_VISIBLE_DEVICE=1 python scripts/process_model.py -N 1000 -r 100 -choose toy -batchs generator_10000 -dataset mnist -size 28 28 1 >out.cout
