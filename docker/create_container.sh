docker run --gpus all \
	-it \
    -h go60 \
    -p 8888:8888 \
    --ipc=host \
    --name mav \
	-v /home/go60/codes/:/workspace/codes \
	mav_img \
	bash           
