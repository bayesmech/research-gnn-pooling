#!/usr/bin/env bash
#SBATCH -n 20
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END

docker stop stochpool_container
docker rm stochpool_container

docker build -t stochpool .
docker run --gpus all --name stochpool_container -d stochpool bash -lic "wandb agent bayesmech/research-gnn-pooling/sse0ki0g"
