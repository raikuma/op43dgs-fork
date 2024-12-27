CUDA_VISIBLE_DEVICES=0 python train.py --eval -s /data/Ricoh360/bricks/imgs/perspective/ -m output/bricks/baseline &&
CUDA_VISIBLE_DEVICES=0 python train.py --eval -s /data/Ricoh360/bridge/imgs/perspective/ -m output/bridge/baseline &&
CUDA_VISIBLE_DEVICES=0 python train.py --eval -s /data/Ricoh360/bridge_under/imgs/perspective/ -m output/bridge_under/baseline

CUDA_VISIBLE_DEVICES=1 python train.py --eval -s /data/Ricoh360/cat_tower/imgs/perspective/ -m output/cat_tower/baseline &&
CUDA_VISIBLE_DEVICES=1 python train.py --eval -s /data/Ricoh360/center/imgs/perspective/ -m output/center/baseline &&
CUDA_VISIBLE_DEVICES=1 python train.py --eval -s /data/Ricoh360/farm/imgs/perspective/ -m output/farm/baseline

CUDA_VISIBLE_DEVICES=2 python train.py --eval -s /data/Ricoh360/flower/imgs/perspective/ -m output/flower/baseline &&
CUDA_VISIBLE_DEVICES=2 python train.py --eval -s /data/Ricoh360/gallery_chair/imgs/perspective/ -m output/gallery_chair/baseline &&
CUDA_VISIBLE_DEVICES=2 python train.py --eval -s /data/Ricoh360/gallery_pillar/imgs/perspective/ -m output/gallery_pillar/baseline

CUDA_VISIBLE_DEVICES=3 python train.py --eval -s /data/Ricoh360/garden/imgs/perspective/ -m output/garden/baseline &&
CUDA_VISIBLE_DEVICES=3 python train.py --eval -s /data/Ricoh360/poster/imgs/perspective/ -m output/poster/baseline

CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python render_equi.py -s /data/Ricoh360/bricks/imgs/perspective/ -m output/bricks/baseline --dataloader --skip_train &&
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python render_equi.py -s /data/Ricoh360/bridge/imgs/perspective/ -m output/bridge/baseline --dataloader --skip_train &&
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python render_equi.py -s /data/Ricoh360/bridge_under/imgs/perspective/ -m output/bridge_under/baseline --dataloader --skip_train

CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python render_equi.py -s /data/Ricoh360/cat_tower/imgs/perspective/ -m output/cat_tower/baseline --dataloader --skip_train &&
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python render_equi.py -s /data/Ricoh360/center/imgs/perspective/ -m output/center/baseline --dataloader --skip_train &&
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python render_equi.py -s /data/Ricoh360/farm/imgs/perspective/ -m output/farm/baseline --dataloader --skip_train

CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python render_equi.py -s /data/Ricoh360/flower/imgs/perspective/ -m output/flower/baseline --dataloader --skip_train &&
CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python render_equi.py -s /data/Ricoh360/gallery_chair/imgs/perspective/ -m output/gallery_chair/baseline --dataloader --skip_train &&
CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python render_equi.py -s /data/Ricoh360/gallery_pillar/imgs/perspective/ -m output/gallery_pillar/baseline --dataloader --skip_train

CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python render_equi.py -s /data/Ricoh360/garden/imgs/perspective/ -m output/garden/baseline --dataloader --skip_train &&
CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python render_equi.py -s /data/Ricoh360/poster/imgs/perspective/ -m output/poster/baseline --dataloader --skip_train

python metrics_equi.py -s /data/Ricoh360/bricks/imgs/ -m output/bricks/baseline &&
python metrics_equi.py -s /data/Ricoh360/bridge/imgs/ -m output/bridge/baseline &&
python metrics_equi.py -s /data/Ricoh360/bridge_under/imgs/ -m output/bridge_under/baseline &&
python metrics_equi.py -s /data/Ricoh360/cat_tower/imgs/ -m output/cat_tower/baseline &&
python metrics_equi.py -s /data/Ricoh360/center/imgs/ -m output/center/baseline &&
python metrics_equi.py -s /data/Ricoh360/farm/imgs/ -m output/farm/baseline &&
python metrics_equi.py -s /data/Ricoh360/flower/imgs/ -m output/flower/baseline &&
python metrics_equi.py -s /data/Ricoh360/gallery_chair/imgs/ -m output/gallery_chair/baseline &&
python metrics_equi.py -s /data/Ricoh360/gallery_pillar/imgs/ -m output/gallery_pillar/baseline &&
python metrics_equi.py -s /data/Ricoh360/garden/imgs/ -m output/garden/baseline &&
python metrics_equi.py -s /data/Ricoh360/poster/imgs/ -m output/poster/baseline
