<div align="center">

## 🎹 [ACMMM '25] Separate to Collaborate: Dual-Stream Diffusion Model for Coordinated Piano Hand Motion Synthesis

🏠 [Homepage](https://monkek123King.github.io/S2C_page)      📄 [Paper](https://arxiv.org/abs/2504.09885)      💾 Dataset [[Google Drive](https://drive.google.com/drive/folders/1JY0zOE0s7v9ZYLlIP1kCZUdNrih5nYEt?usp=sharing)]/[[Hyper.ai](https://hyper.ai/datasets/32494)]/[[Zenodo](https://zenodo.org/records/13297386)]      🤗 Model [[HuggingFace](https://huggingface.co/thuteam/S2C/tree/main)]

</div>

-----

### 📢 News

  * **`Sept 2025`:** Experiment checkpoints are released [here](https://huggingface.co/thuteam/S2C)\! 🎉
  * **`July 2025`:** Our paper has been accepted to ACMMM 2025\! 🥳
  * **`April 2025`:** The paper is now available on [arXiv](https://arxiv.org/abs/2504.09885). ☕️

-----

## 🚀 Getting Started

### 🔧 Installation

**a. Create a conda virtual environment and activate it.**
```shell
conda create -n S2C python=3.10 -y
conda activate S2C
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
```

**c. Clone S2C.**
```
git clone https://github.com/monkek123King/S2C.git
```

**d. Install other requirements.**
```shell
cd S2C
pip install -r requirement.txt
```

**e. Prepare MANO models.**

Besides, you also need to download the MANO model. Please visit the [MANO website](https://mano.is.tue.mpg.de/) and register to get access to the downloads section. We only require the right hand model. You need to put MANO_RIGHT.pkl under the ./mano folder.

**f. Prepare pretrained models. (Used in training.)**

Download pretrained HuBert([Large](https://huggingface.co/facebook/hubert-large-ls960-ft))  to `S2C/checkpoints`.

**g. Prepare Gesture Autoencoder model. (Used in evaluation.)**

Download pretrained [Gesture Autoencoder model](https://drive.google.com/file/d/1G2Fe_zlJn8I_U_VGldH4SsIa_KauvG3p/view?usp=sharing) to `S2C/checkpoints`

```
checkpoints
├── gesture_autoencoder_checkpoint_best.bin
├── hubert-large-ls960-ft/
```

### 📦 Prepare Dataset

**PianoMotion10M**

Download PianoMotion10M V1.0 full dataset data [HERE](https://drive.google.com/drive/folders/1JY0zOE0s7v9ZYLlIP1kCZUdNrih5nYEt?usp=sharing).

```
cd /path/to/PianoMotion10M_Dataset
unzip annotation.zip
unzip audio.zip
unzip midi.zip
```


**Folder structure**
```
/path/to/PianoMotion10M_Dataset
├── annotation/
│   ├── 1033685137/
│   │   ├── BV1f34y1i7U1/
│   │   │   ├──BV1f34y1i7U1_seq_0000.json
│   │   │   ├──BV1f34y1i7U1_seq_0001.json
│   │   │   ├──...
│   │   ├── BV1X44y1J7CR/
│   ├── 2084102325/
│   ├── ...
├── audio/
│   ├── 1033685137/
│   │   ├── BV1f34y1i7U1/
│   │   │   ├──BV1f34y1i7U1_seq_0000.mp3
│   │   │   ├──BV1f34y1i7U1_seq_0001.mp3
│   │   │   ├──...
│   │   ├── BV1X44y1J7CR/
│   ├── 2084102325/
│   ├── ...
├── midi/
│   ├── 1033685137/
│   │   ├── BV1f34y1i7U1.mid
│   │   ├── BV1X44y1J7CR.mid
│   │   ├── ...
│   ├── 2084102325/
│   ├── ...
├── train.txt
├── test.txt
├── valid.txt
```

**Usage**

`draw.py` shows the usage of our dataset and visualizes some samples of hand motions under `./draw_sample`.

```shell
python draw.py
```

### 🏋️ Train and Evaluate

**Please ensure you have prepared the environment and the PianoMotion10M dataset.**

**Train and Test**

Train S2C Position Predictor with Hubert and transformer. Feel free to change audio feature extractor by `--wav2vec_path`. The result will be stored in `./logs/`.
```
python train.py --experiment_name piano2posi_LR --bs_dim 6 --adjust --is_random --up_list 1467634 66685747 \
--data_root ./ --iterations 200000 --batch_size 8 --train_sec 8 --feature_dim 512 \
--wav2vec_path ./checkpoints/hubert-large-ls960-ft --check_val_every_n_iteration 1000 --save_every_n_iteration 1000 \
--latest_layer tanh --encoder_type transformer --num_layer 4
```

Train S2C Gesture Generator with Hubert and transformer. The result will be stored in `./logs/`.
```
python train_diffusion.py --experiment_name piano2mot --is_random --unet_dim 256 --iterations 800000 \
--bs_dim 96  --batch_size 16 --train_sec 8 --data_root ./ \
--xyz_guide  --check_val_every_n_iteration 1000 --save_every_n_iteration 1000 \
--adjust --piano2posi_path logs/piano2posi_LR --encoder_type transformer --num_layer 4 \
--lr 1e-5 --fusion 4 --obj pred_v
```

Eval S2C after training S2C Gesture Generator on the validation set.
```
python eval.py --exp_path /path/to/logs (e.g. ./logs/piano2mot)  --data_root /path/to/PianoMotion10M_Dataset --valid_batch_size 64 --mode valid
```

**Visualization**

Visualize the results, which will be stored in `./results`.

```
python infer.py --exp_path /path/to/logs --data_root /path/to/PianoMotion10M_Dataset --valid_batch_size 64 --mode valid
```

-----

## ✍️ Citation

If you find our work useful for your research, please consider citing our paper and giving this repository a star 🌟.

```bibtex
@inproceedings{liu2025separate,
  title={Separate to Collaborate: Dual-Stream Diffusion Model for Coordinated Piano Hand Motion Synthesis},
  author={Liu, Zihao and Ou, Mingwen and Xu, Zunnan and Huang, Jiaqi and Han, Haonan and Li, Ronghui and Li, Xiu},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
  pages={9743--9752},
  year={2025}
}
```
