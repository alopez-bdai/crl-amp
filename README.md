Implementation of Adversarial Motion Priors ([AMP](https://arxiv.org/abs/2104.02180)) for Isaacgym environments.

Maintained by [Fatemeh Zargarbashi](mailto:fatemeh.zargarbashi@inf.ethz.ch).

# Install

Python3.8 and Poetry
```bash
curl -sSL https://install.python-poetry.org | python3 - --version 1.4.0
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc 
sudo apt-get install python3.8 build-essential python3.8-dev python3.8-distutils 
```
Download [issacgym preview 4](https://developer.nvidia.com/isaac-gym/download) and place it in the root directory, such that `isaacgym` is a subdirectory of `crl-amp`.

```bash
cd crl-amp
poetry env use /usr/bin/python3.8
poetry install
```

# Run on a server using docker
1. tmux
2. clone the git repo and isaacgym repo
```bash
cd crl-amp
cd dockerfiles
docker build -t crl_amp_img .
```
3. make sure to change <your_path>
```bash
docker run -v /local/home/<your_path>/crl-amp:/home/crl-amp --name crl_amp --runtime=nvidia --gpus "device=1" -it crl_amp_img bash
```

4. now you should be in a shell
```bash
cd home/crl-amp
poetry install
poetry run python amp/scripts/train.py --task=go1_amp --dv --wb --dr
```

# Acknowledgements

This project contains code inspired by and/or incorporates portions of code from the following repositories:
- [AMP for hardware](https://github.com/Alescontrela/AMP_for_hardware)