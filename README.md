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

The result is a tab-delimited text file with extension `.deepnovo_denovo`. Each row shows the predicted sequence, confidence score, and other information for a feature provided in the input feature_file.

**Step 2:** Test de novo sequencing results on labeled features:

    deepnovo_main --test --target_file <target_file> --predicted_file <predicted_file>

For example:

    --target_file plasma/testing_plasma.feature.csv
    --predicted_file plasma/testing_plasma.feature.csv.deepnovo_denovo
    
As this testing feature_file is labeled, it includes the target sequence for each feature (column `seq`). Thus, DeepNovo can compare the predicted sequence to the target sequence and calculate the accuracy. The result includes 3 files. The file with extension `.accuracy` shows the comparison result for each feature. The other 2 files can be ignored.
