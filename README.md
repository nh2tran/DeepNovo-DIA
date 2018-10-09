# DeepNovo-DIA

## Deep learning enables de novo peptide sequencing from DIA.

- Publication: under review.
- Data and pre-trained model on MassIVE repository: ftp://massive.ucsd.edu/MSV000082368/other/
- Knapsack matrix: https://drive.google.com/open?id=1aHDGphyzTo2hMMXlwkLCRVDp9JB9ph34
- We provide a Linux pre-compiled file `deepnovo_main`, which can be used to train a model, to perform de novo sequencing and to test the accuracy.
We have packed the cpu-version of TensorFlow and other required Python libraries, so the software can run on any Linux machine.
This version requires two input files: a spectrum mgf file and a feature csv file.
- We also provide a Windows executable file, which can be downloaded from the authors' website: https://cs.uwaterloo.ca/~mli/index.html
This version includes feature detection and pre-processing modules to run directly on `.raw` files (from Thermo instruments).
This version also has a graphic user interface for convenience.

(backup link: https://drive.google.com/open?id=1T07-YHvJdmSE1emx8U8YmYrtq0Z1mEbN)

If you have any problems running our software, please contact hieutran1985@gmail.com.

### How to use DeepNovo?

For Windows version, please refer to the documentation file.

For Linux version, please see the following instructions.

**Step 1:** Run de novo sequencing with a pre-trained model:

    deepnovo_main --search_denovo --train_dir <training_folder> --denovo_spectrum <spectrum_file> --denovo_feature <feature_file>

We have provided a pre-trained folder in the above repository together with three testing datasets. For example:

    --train_dir train.urine_pain.ioncnn.lstm
    --denovo_spectrum plasma/testing_plasma.spectrum.mgf
    --denovo_feature plasma/testing_plasma.feature.csv

The result is a tab-delimited text file with extension `.deepnovo_denovo`. 
Each row shows the predicted sequence, confidence score, and other information for a feature provided in the input feature_file.

Note that the feature_file can be labeled or unlabeled. 
If labeled, the target sequence of each feature is provided in column `seq`, and we can run the next step to calculate accuracy. 
If not labeled, the column `seq` is simply empty.

**Step 2:** Test de novo sequencing results on labeled features:

    deepnovo_main --test --target_file <target_file> --predicted_file <predicted_file>

For example:

    --target_file plasma/testing_plasma.feature.csv
    --predicted_file plasma/testing_plasma.feature.csv.deepnovo_denovo
    
As this testing feature_file is labeled, it includes the target sequence for each feature. 
Thus, DeepNovo can compare the predicted sequence to the target sequence and calculate the accuracy. 
The result includes 3 files. The file with extension `.accuracy` shows the comparison result for each feature. 
The other 2 files can be ignored. The accuracy summary is also printed out to the terminal.

**Step 3:** Train a new model:

    deepnovo_main --train --train_dir <training_folder> --train_spectrum <train_spectrum_file> --train_feature <train_feature_file> --valid_spectrum <valid_spectrum_file> --valid_feature <valid_feature_file>

In order to train a new model, you will need a training set and a validation set, each including a spectrum mgf file and a feature csv file. 

