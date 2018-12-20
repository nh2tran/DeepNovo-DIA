# DeepNovo-DIA

- Publication: Deep learning enables de novo peptide sequencing from DIA mass spectrometry. Nature Methods, 2018. (https://www.nature.com/articles/s41592-018-0260-3)
- Data and pre-trained model on MassIVE repository: ftp://massive.ucsd.edu/MSV000082368/other/
- Knapsack matrix: https://drive.google.com/open?id=1aHDGphyzTo2hMMXlwkLCRVDp9JB9ph34
- We provide a Linux pre-compiled file `deepnovo_main`, which can be used to train a model, to perform de novo sequencing and to test the accuracy.
We have packed the cpu-version of TensorFlow and other required Python libraries, so the software can run on any Linux machine.
This version requires two input files: a spectrum mgf file and a feature csv file.
The knapsack matrix (available from the above link) needs to be in the same folder as `deepnovo_main`. 
- We also provide a Windows executable file, which can be downloaded from the authors' website: https://cs.uwaterloo.ca/~mli/index.html
This version includes feature detection and pre-processing modules to run directly on `.raw` files (e.g., from Thermo instruments).

(backup link: https://drive.google.com/open?id=1T07-YHvJdmSE1emx8U8YmYrtq0Z1mEbN)

## How to use DeepNovo?

For Windows version, please refer to the documentation file.

For Linux version, please see the following instructions.

### Step 1: Run de novo sequencing with a pre-trained model:

    deepnovo_main --search_denovo --train_dir <training_folder> --denovo_spectrum <spectrum_file> --denovo_feature <feature_file>

We have provided a pre-trained folder in the above repositories together with three testing datasets. For example:

    --train_dir train.urine_pain.ioncnn.lstm
    --denovo_spectrum plasma/testing_plasma.spectrum.mgf
    --denovo_feature plasma/testing_plasma.feature.csv

The mgf file contains all MS/MS spectra in the dataset.
Each spectrum starts with the line "BEGIN IONS", followed by 5 header lines:
- "TITLE": not relevant
- "PEPMASS": the center of DIA m/z window of the spectrum
- "CHARGE": not relevant
- "SCANS": the MS/MS scan id. For example, "F1:3" means scan number 3 of fraction 1.
- "RTINSECONDS": the retention time of the spectrum

After those headers lines comes the list of pairs of (m/z, intensity) of fragment ions in the spectrum.
Finally, the spectrum ends with the line "END IONS"

The csv file contains all precursor features detected from LC/MS map.
Each feature includes the following columns:
- "spec_group": the feature id. For example, "F1:6427" means feature number 6427 of fraction 1.
- "m/z": the mass-to-charge ratio
- "z": the charge
- "rt_mean": the mean of the retention time range
- "seq": the column is empty when running de novo sequencing. 
In training/testing modes, it contains the target peptide sequence.
- "scans": list of all MS/MS spectra collected for the feature so that 
they are within the feature’s retention time range and their DIA m/z windows must cover the feature’s m/z.
The spectra’s ids are separated by semicolon. 
The spectra’s ids can be used to locate the spectra in the mgf file.
- "profile": intensity values over the retention time range. 
The values are pairs of "time:intensity" and are separated by semicolon. 
The time points align to the time of spectra in the column "scans".
- "feature_area": precursor feature area estimated by the feature detection.

The result is a tab-delimited text file with extension `.deepnovo_denovo`. 
Each row includes the following columns:
- feature_id
- feature_area
-	predicted_sequence
- predicted_score
- predicted_position_score: positional score for each amino acid in predicted_sequence
- precursor_mz
- precursor_charge
- protein_access_id: not relevant
- scan_list_original: list of scan ids of DIA spectra associated with this feature
- scan_list_middle: list of DIA spectra used for de novo sequencing
- predicted_score_max: same as predicted_score, not relevant

### Step 2: Test de novo sequencing results on labeled features:

    deepnovo_main --test --target_file <target_file> --predicted_file <predicted_file>

For example:

    --target_file plasma/testing_plasma.feature.csv
    --predicted_file plasma/testing_plasma.feature.csv.deepnovo_denovo
    
As the testing feature file is labeled, it includes the target sequence for each feature. 
Thus, DeepNovo can compare the predicted sequence to the target sequence and calculate the accuracy. 
The result includes 3 files. The file with extension `.accuracy` shows the comparison result for each feature. 
The other 2 files can be ignored. The accuracy summary is also printed out to the terminal.

### Step 3: Train a new model:

    deepnovo_main --train --train_dir <training_folder> --train_spectrum <train_spectrum_file> --train_feature <train_feature_file> --valid_spectrum <valid_spectrum_file> --valid_feature <valid_feature_file>

In order to train a new model, you will need a training set and a validation set, each including a spectrum mgf file and a feature csv file. 

