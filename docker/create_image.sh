
image_name=textssl_img
docker build -t $image_name . --build-arg UNAME=$(whoami) \
                               --build-arg UID=$(id -u) \
                               --build-arg GID=$(id -g)