# DeepNovo-DIA

## Deep learning enables de novo peptide sequencing from DIA.

- Publication: under review.
- Software and Data repository: https://drive.google.com/open?id=0By9IxqHK5MdWalJLSGliWW1RY2c

### How to use DeepNovo?

**Step 1:** Run de novo sequencing with a pre-trained model:

    deepnovo_main --search_denovo --train_dir <training_folder> --denovo_spectrum <spectrum_file> --denovo_feature <feature_file>

We have provided a pre-trained folder in the above repository together with three testing datasets. For example:

    --train_dir train.urine_pain.ioncnn.lstm
    --denovo_spectrum plasma/testing_plasma.spectrum.mgf
    --denovo_feature plasma/testing_plasma.feature.csv
