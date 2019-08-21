model="mnist"
for epoch in generator_1000 generator_2000 generator_3000 generator_4000 generator_5000 generator_6000 generator_7000 generator_8000
do
    CUDA_VISIBLE_DEVICE=2 python scripts/process_model.py -N 1500 -r 100 -choose toy -batchs $epoch -dataset $model -size 28 28 1
done
#CUDA_VISIBLE_DEVICE=2 python scripts/process_model.py -N 1000 -r 500 -choose toy -batchs generator_2000 -dataset $model -size 64 64 1
#CUDA_VISIBLE_DEVICE=0 python scripts/process_model.py -N 1000 -r 500 -choose toy -batchs generator_3000 -dataset $model -size 28 28 1 >out.cout
#CUDA_VISIBLE_DEVICE=0 python scripts/process_model.py -N 1000 -r 500 -choose toy -batchs generator_5000 -dataset $model -size 28 28 1 >out.cout
#CUDA_VISIBLE_DEVICE=0 python scripts/process_model.py -N 1000 -r 500 -choose toy -batchs generator_7000 -dataset $model -size 28 28 1 >out.cout
#CUDA_VISIBLE_DEVICE=0 python scripts/process_model.py -N 1000 -r 500 -choose toy -batchs generator_10000 -dataset $model -size 28 28 1 >out.cout
