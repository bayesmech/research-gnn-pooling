FROM pytorch/pytorch:latest

WORKDIR /usr/work/research-gnn-poooling
COPY stochpool stochpool
COPY datasets datasets

RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.0+cu116.html
RUN pip install torch-sparse -f https://data.pyg.org/whl/torch-1.12.0+cu116.html
RUN pip install torch-geometric
RUN python -m pip install tqdm
RUN python -m pip install wandb

ENV PYTHONPATH="/usr/work/research-gnn-poooling"
ENV WANDB_API_KEY="0cc84f7b0a7b22052b6ddc033a3128a589005f79"
