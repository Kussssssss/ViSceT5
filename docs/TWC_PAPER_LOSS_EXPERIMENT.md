# Joint VQA + TWC with the paper loss

Branch `tmp2` starts from `efdd1e3`, before the two-stage pretrain/VQA change.
It retains joint VQA + TWC training and the encoder layout
`[question, image, Origin, Related]`. The objective change is
`twc_loss: paper_bce`: cosine similarity divided by temperature, row softmax
over valid candidates, then symmetric binary cross entropy against the
unchanged soft labels (0, 0.9, 1). Padding is excluded. The legacy `bce` and
`infonce` modes remain available for existing checkpoints.

Temperature is now fixed at 0.2 (`twc_temperature_trainable: false`), stored as
checkpoint buffers with no gradient, optimizer entry, or weight decay. Other
settings stay as in the base commit: augmentation 0.35, TWC weight 1.0,
batch size 32, model LR 1e-4, and five epochs.
The notebook retains its original `qavit_mode=no_question`;
the YAML retains `qavit_mode=off`. Match the mode to the baseline being compared.

`notebooks/twc_colab2.ipynb` and the Colab launcher select `tmp2`. New output and
Hub names include `paperbce` and `fixed`, avoiding automatic resume from sigmoid-BCE
or learnable-temperature runs.
Optional smoke/parity checks are disabled by default. The main notebook run is
full finetuning, with the existing per-epoch Hub checkpoint uploads and final
upload. It is not the two-stage runner on branch `tmp`.
