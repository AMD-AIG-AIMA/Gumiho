# Gumiho vLLM Integration

This repository provides updated code for integrating Gumiho with vLLM.

## Usage

Download a publicly available Docker image, and then replace the vLLM library code in the Docker with the code from our folder.

1. Download the public Docker image:

   ```
   docker pull rocm/vllm:rocm6.3.1_instinct_vllm0.8.3_20250415
   ```

2. Run the container to create the environment:

   ```
   docker run -d \
     --privileged \
     --device=/dev/kfd \
     --device=/dev/dri \
     --network=host \
     --group-add video \
     --ipc=host \
     --cap-add=SYS_PTRACE \
     --security-opt seccomp=unconfined \
     --shm-size 32G \
     --name Gumiho_vLLM \
     -v [PATH-TO-GUMIHO]:/Gumiho \
     rocm/vllm:rocm6.3.1_instinct_vllm0.8.3_20250415 \
     /bin/bash
   ```

   **Note:** Replace `[PATH-TO-GUMIHO]` with the actual path to your Gumiho folder on the host machine.

3. Copy the `*.so` files from the Docker's `/usr/local/lib/python3.12/dist-packages/vllm` folder to `/Gumiho/vllm/` (execute this inside the running container):

   ```
   docker exec -it Gumiho_vLLM /bin/bash
   cp /usr/local/lib/python3.12/dist-packages/vllm/*.so /Gumiho/vllm/
   ```

4. Replace the vLLM folder in the Docker with `/Gumiho/vllm` (continue inside the container):

   ```
   rm -rf /usr/local/lib/python3.12/dist-packages/vllm
   cp -r /Gumiho/vllm /usr/local/lib/python3.12/dist-packages/
   ```

5. Run the script (still inside the container):

   ```
   cd /Gumiho
   bash script/vllm.sh
   ```
