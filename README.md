# MotionCompass
## Paper

[MotionCompass: Language-Guided Semantic Steering for Stochastic Human Motion Prediction](https://doi.org/10.3390/electronics15235446)

## Code Release

**The preliminary code for MotionCompass is now available!**

This version has been tested and runs successfully, but it may still contain redundant experimental and testing code. 
We will refine the code structure to make the core implementation of our method clearer and gradually add citations, source attributions, and related documentation.
Besides, the preliminary release uses WAT as the baseline. Code using ATACB as the baseline will be added in a future update.

If you encounter any issues, please open an Issue. We will address them as soon as possible.

Please refer to the [Quick Start (Preliminary Release)](#quick-start-preliminary-release) section to run the code.

## TODO

- [ ] Release code for ATACB-baseline MotionCompass.
- [ ] Release pretrained weights.
- [ ] Complete the README with environment setup, training, evaluation instructions and so on.
- [ ] Refine the code structure and remove redundant testing code.
- [ ] Add citations, source attributions, and related documentation.

## Quick Start (Preliminary Release)

Please refer to the run script for your target dataset, such as `run_ntu.sh`.

To run the code successfully, you may also need to download the datasets and pretrained classifiers used for evaluation. Please refer to [WAT](https://github.com/wei-mao-2019/WAT) for the relevant instructions.

Additionally, download [FacebookAI/roberta-base](https://huggingface.co/FacebookAI/roberta-base) from Hugging Face and place it under `WAT-base/FacebookAI/roberta-base/`:

```text
MotionCompass/
└── WAT-base/
    ├── models/
    ├── ...
    └── FacebookAI/
        └── roberta-base/
            └── ... (model and tokenizer files)
```

We will provide more detailed setup and usage instructions as soon as possible.

## Acknowledgements

Our implementation builds upon [WAT](https://github.com/wei-mao-2019/WAT) and draws on [ATACB](https://github.com/hyqlat/ATACB). We sincerely thank the authors for making their code publicly available and for their valuable contributions to stochastic human motion prediction.
