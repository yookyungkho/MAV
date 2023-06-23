docker run --gpus all \
	-it \
    -h go60 \
    -p 628:8888 \
    --ipc=host \
    --name textssl \
	-v /home/go60/projects/:/workspace/codes \
	textssl_img \
	bash                           