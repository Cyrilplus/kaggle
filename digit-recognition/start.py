exec_file=lenet5.py
docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all --rm -w="/root" -v $PWD:/root paddlepaddle/paddle:latest-gpu python $exec_file