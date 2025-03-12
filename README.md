
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

Assuming Debian 10

```sh
sudo bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"
sudo apt-get install clang-18 lldb-18 lld-18
wget https://github.com/Kitware/CMake/releases/download/v3.31.6/cmake-3.31.6-linux-x86_64.sh
chmod +x cmake-3.31.6-linux-x86_64.sh
echo 'export PATH="/home/viet.ly/cmake-3.31.6-linux-x86_64/bin:$PATH"' >> ~/.profile
echo 'export PATH="/usr/local/cuda-12.4/bin:$PATH"' >> ~/.profile

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
run cmake script then update env to point to latest cmake
```

wget https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo add-apt-repository contrib
sudo apt-get update
sudo apt-get -y install cuda=12.4.0-1
```

update lcudart
```
export LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64/:$LD_LIBRARY_PATH"
```


