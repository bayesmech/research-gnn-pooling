FROM pytorch/pytorch:latest

WORKDIR /usr/work/research-gnn-poooling
COPY stochpool stochpool
COPY datasets datasets

RUN python -m pip install torch-scatter torch-sparse torch-cluster torch-spline-conv  \
    torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cpu.html
RUN python -m pip install -r requirements.txt

CMD python -m stochpool --model stochpool --dataset proteins --epochs 2
