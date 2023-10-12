# Distributed Deep Joint Source-Channel Coding over a Multiple Access Channel \[[Paper](https://arxiv.org/pdf/2211.09920.pdf)\]

## Citation
Please cite the following paper if this code or paper has been useful to you:
```
@inproceedings{yilmaz2023distributed,
      title={Distributed Deep Joint Source-Channel Coding over a Multiple Access Channel}, 
      author={Selim F. Yilmaz and Can Karamanlı and Deniz Gündüz},
      booktitle={2023 IEEE International Conference on Communications (ICC)}, 
      year={2023},
      organization={IEEE}
}
```

## Abstract

We consider distributed image transmission over a noisy multiple access channel (MAC) using deep joint source-channel coding (DeepJSCC). It is known that Shannon's separation theorem holds when transmitting independent sources over a MAC in the asymptotic infinite block length regime. However, we are interested in the practical finite block length regime, in which case separate source and channel coding is known to be suboptimal. We introduce a novel joint image compression and transmission scheme, where the devices send their compressed image representations in a non-orthogonal manner. While non-orthogonal multiple access (NOMA) is known to achieve the capacity region, to the best of our knowledge, non-orthogonal joint source channel coding (JSCC) scheme for practical systems has not been studied before. Through extensive experiments, we show significant improvements in terms of the quality of the reconstructed images compared to orthogonal transmission employing current DeepJSCC approaches particularly for low bandwidth ratios. We publicly share source code to facilitate further research and reproducibility.

### Installation

To install environment, run the following commands after installing `torch>=1.10.0` for your specific training device:

```
git clone https://github.com/ipc-lab/deepjscc-noma.git
cd deepjscc-noma
pip install -r requirements.txt
```

### Reproducing the Experiments on the Paper

To reproduce the trainings of the methods in the paper for all figures, the following command can be used:

```
cd src
python train.py experiment=<experiment_file>
```

In the command above, `<experiment_file>` variable is can be modified with the following files (in configs/experiment folder and it uses the default hyperparameters specified in the configs):

- `main_comparison_TDMA_C3.yaml`: DeepJSCC-TDMA for 1/3 bandwidth ratio
- `main_comparison_NOMA_C3.yaml`: DeepJSCC-NOMA for 1/3 bandwidth ratio
- `main_comparison_C3_SingleUser.yaml`: DeepJSCC-SingleUser for 1/3 bandwidth ratio
- `main_comparison_NOMA_CL_C3.yaml`: DeepJSCC-NOMA-CL for 1/3 bandwidth ratio (run with additional argument for checkpoint `model.net.ckpt_path=<path of best Deepjscc-SingleUser model>`)
- `fairness_C3.yaml`: DeepJSCC-NOMA-CL for 1/3 banwidth ratio in fairness plot
- `main_comparison_TDMA_C6.yaml`: DeepJSCC-TDMA for 1/6 bandwidth ratio
- `main_comparison_NOMA_C6.yaml`: DeepJSCC-NOMA for 1/6 bandwidth ratio
- `main_comparison_C6_SingleUser.yaml`: DeepJSCC-SingleUser for 1/6 bandwidth ratio
- `main_comparison_NOMA_CL_C6.yaml`: DeepJSCC-NOMA-CL for 1/6 bandwidth ratio (run with additional argument for checkpoint `model.net.ckpt_path=<path of best Deepjscc-SingleUser model>`)
- `fairness_C6.yaml`: DeepJSCC-NOMA-CL for 1/6 banwidth ratio in fairness plot

### See Hyperparameters and Configs

To see hyperparameters and configs, run the following commands:

```
cd src
python train.py --help
```

### Training a model

To train a new model, run the following command after replacing `<hyperparameters>` the desired hyperparameters:

```
cd src
python train.py <hyperparameters>
```

### Notes

This code is based on Pytorch Lightning and Hydra. We use [Lightning Hydra Template](https://github.com/ashleve/lightning-hydra-template/) as our base code. For more details on the template we use, please see the [README.md](https://github.com/ashleve/lightning-hydra-template/blob/main/README.md) of the template.
