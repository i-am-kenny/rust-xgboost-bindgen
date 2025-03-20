
```
git submodule update --init --recursive
```

```
brew install libomp
```

Maybe?
```
conda install -c conda-forge py-xgboost
```

## CUDA Support

Assuming Debian 10. For other installations, refer to https://docs.nvidia.com/cuda/archive/12.4.0/cuda-installation-guide-linux/index.html#package-manager-installation

```sh
sudo bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"
sudo apt-get install clang-18 lldb-18 lld-18

wget https://github.com/Kitware/CMake/releases/download/v3.31.6/cmake-3.31.6-linux-x86_64.sh
chmod +x cmake-3.31.6-linux-x86_64.sh
./cmake-3.31.6-linux-x86_64.sh

echo 'export PATH="$HOME/cmake-3.31.6-linux-x86_64/bin:$PATH"' >> ~/.bashrc

wget https://developer.download.nvidia.com/compute/cuda/repos/debian10/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo add-apt-repository contrib
sudo apt-get update
sudo apt-get install linux-headers-$(uname -r)

sudo apt-get -y install cuda-12-4
sudo apt-get install -y nvidia-kernel-open-dkms
sudo apt-get install -y cuda-drivers

echo 'export PATH="/usr/local/cuda-12.4/bin:$PATH"' >> ~/.bashrc
echo 'export CUDA_LIBRARY_PATH="/usr/local/cuda-12.4/lib64/"' >> ~/.bashrc

```

### Optional
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

rustup component add rust-analyzer
```


