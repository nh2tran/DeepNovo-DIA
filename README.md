# DeepNovo-DIA

## Deep learning enables de novo peptide sequencing from DIA.

- Publication: under review.
- Software and Data repository: https://drive.google.com/open?id=0By9IxqHK5MdWalJLSGliWW1RY2c

### How to use DeepNovo?

**Step 1:** Run de novo sequencing with a pre-trained model

    deepnovo_main --search_denovo --train_dir <training_folder> --denovo_spectrum <spectrum_file> --denovo_feature <feature_file>
