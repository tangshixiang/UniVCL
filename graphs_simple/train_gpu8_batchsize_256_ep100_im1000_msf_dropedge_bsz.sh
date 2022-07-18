# source /mnt/lustre/share/spring/s0.3.3
set -x
export LC_ALL=en_US

NTASKS=${NTASKS:-1}
JOBTYPE=${JOBTYPE:-$2}
DROPOUT_RATE=${DROPOUT_RATE:-$3}
DIFFUSION_METHOD_T=${DIFFUSION_METHOD_Q:-$4}
GPUs=${GPUs:-8}
EPOCHs=${EPOCHs:-100}
BATCHSIZE=${BATCHSIZE:-256}
LRBASE=${LRBASE:-0.05}
DATASET=${DATASET:-"imagenet"}
GCN_configs=${GCN_configs:-$1}
BASENAME_GCN_configs="$(basename -- $GCN_configs)"
CHECKPOINT_PATH=experiments/graphs_simple_bsz/msf_dropedge_checkpoints_diffmethodT_${DIFFUSION_METHOD_T}_dropoutrate${DROPOUT_RATE}_g${GPUs}_ep_${EPOCHs}_bs_${BATCHSIZE}_ds_${DATASET}_cfg_${BASENAME_GCN_configs}
COMPLETEFILE=${CHECKPOINT_PATH}/complete.txt

LR=$(echo "scale=10; ${LRBASE}*${BATCHSIZE}/256.0"|bc)

while [ ! -f $COMPLETEFILE ];
do
  spring.submit run --mpi=pmi2 -p spring_scheduler --job-type ${JOBTYPE} --job-name=R-SC210077.00107 -n${NTASKS} --gres=gpu:${GPUs} --ntasks-per-node=${GPUs}  \
  "python -u trainers/train_graph_simple_bsz_dropnodes.py \
    --cos \
    --weak_strong \
    --learning_rate ${LR} \
    --epochs ${EPOCHs} \
    --arch resnet50 \
    --topk 10 \
    --dropout_rate ${DROPOUT_RATE} \
    --momentum 0.99 \
    --mem_bank_size 128000 \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --batch_size ${BATCHSIZE} \
    --dataset ${DATASET} \
    --GCN_configs ${GCN_configs} \
    --diffusion_method_t ${DIFFUSION_METHOD_T} \
    /mnt/lustre/share/tangshixiang/data/ImageNet/images"
  sleep 1m
done

# while [ ! -f $COMPLETEFILE ];
# do
#   spring.submit run --mpi=pmi2 -p spring_scheduler --job-name=R-SC210077.00107 -n${NTASKS} --gres=gpu:${GPUs} --ntasks-per-node=${GPUs}  \
#   "python -u train_graph_simple_recheck.py \
#     --cos \
#     --weak_strong \
#     --learning_rate ${LR} \
#     --epochs ${EPOCHs} \
#     --arch resnet50 \
#     --topk 10 \
#     --momentum 0.99 \
#     --mem_bank_size 128000 \
#     --checkpoint_path ${CHECKPOINT_PATH} \
#     --batch_size ${BATCHSIZE} \
#     --dataset ${DATASET} \
#     --GCN_configs ${GCN_configs} \
#     /mnt/lustre/share/tangshixiang/data/ImageNet/images"
#   sleep 1m
# done
