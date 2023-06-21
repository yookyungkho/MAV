docker run --gpus all \
	-it \
    -h jaegeekim \
    -p 628:8888 \
    --ipc=host \
    --name jaehee_textssl \
	-v /home/jaegeekim/projects/:/workspace/codes \
	jaehee_textssl_img \
	bash                           