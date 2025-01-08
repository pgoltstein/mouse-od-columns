# Mouse Ocular Dominance Columns: Data, analysis and figures

This repository contains code for analyzing data and producing figure panels as presented in the manuscript "A column-like organization for ocular dominance in mouse visual cortex" by Pieter M. Goltstein, David Laubender, Tobias Bonhoeffer and Mark Hübener (Max Planck Institute for Biological Intelligence).  

A detailed instruction can be found below. In case of any questions, please do not hesitate to contact us.

---

### Part 0: Setting up the analysis environment

#### Part 0a: Install programs/tools

1. Download and install Python 3.8.16  (Anaconda distribution).  
_Follow instructions on https://anaconda.org_

2. Download and install “Git” (this is optional, the code can also be downloaded manually).  
_Follow instructions on https://git-scm.com_

3. Download and install the “GIN command line client” (this is optional, the data can also be downloaded manually).  
_Follow instructions on https://gin.g-node.org (under the 'Help' tab)_

#### Part 0b: Download the data

1. Open up any type of command line shell (csh, bash, zsh, cmd.exe, etc) and change the current directory to a drive/folder in which you like to have the entire project. The dataset is approximately 47 GB, but seems to also have a large .git folder when downloaded automatically, so make sure you have enough free disk space (~100GB).

2. Log in on the gin server  
``` gin login ```  

3. Download the 'small file' dataset  
``` gin get pgoltstein/mouse-od-columns ```  

4. Download the 'large files' from within the repository folder
```
cd mouse-od-columns
gin get-content
```  
Note that this can take some time as it downloads about 47 GB of data.

Or, if you prefer, you can instead download the data manually from gin.g-node  
https://gin.g-node.org/pgoltstein/mouse-od-columns  

#### Part 0c: Download the code

1. CD into the repo-folder (called “mouse-od-columns”) if you are not already there ..

2. Download the code (this will be placed in a newly created subfolder “code”)  
``` git clone https://github.com/pgoltstein/mouse-od-columns.git code ```

3. Add a folder for the figure output  
``` mkdir figureout ```

#### Part 0d: Check if you have everything

Your directory structure should look like this:
```
- mouse-od-columns (or any other name you chose for your base directory)
  - code
    - part1-od-clusters-layer4
      - make-response-maps.py
      - ... etc
    - part2-od-columns
    - ... etc
  - data
    - suite2pdata-layer4
    - suite2pdata-layer2345
    - ...
  - figureout
  - ... etc
```

#### Part 0c: Create analysis environment

1. Create the python environment from the (system dependent) yaml file in the code folder  
``` conda env create -f ./code/environment_general.yaml --name mouseodcolumns ```   
or try  
``` conda env create -f ./code/environment_windows.yaml --name mouseodcolumns ```   
or  
``` conda env create -f ./code/environment_macosx.yaml --name mouseodcolumns ```

2. Activate the environment  
```conda activate mouseodcolumns ```

3. All code should be run from the respective "code path", that is, cd into the code directory and run it using python. This is for the reason that it (by defaut) will look for the data in a reletive path starting from the folder where the python code is stored. So, for example, to make an ODI map as in figure 1a:
```
cd code
cd part1-od-clusters-layer4
python make-response-maps.py O03 2 -m 20 --examplecells 247 1 6
```
The maps will be saved in the folder called "figureout"

---

### Part 1: Clusters for Ocular Dominance in cortical layer 4

These paths are just FYI:  
Code path: “./code/part1-od-clusters-layer4”  
Data paths: “./data/part1-responsemapdata”, “./data/part1-planedata-od-layer4”, “./data/part1-planedata-retinotopy-layer4”, “./data/part1-planedata-od-layer4-caiman”, “./data/part1-tileddata-od-layer4”, “./data/part1-tileddata-ret-layer4”  
Processed data path: “./data/part1-processeddata"  

Note: There is already processed data supplied in the repository, however, in case you like to rerun it, the code below will recreate the processed data

Data processing for 1g-i:
*	``` python process-layer4-clusters-across-mice.py b6-gcamp6s ```
*	``` python process-layer4-clusters-across-mice.py b6-gcamp6s -sh ```
*	``` python process-layer4-clusters-across-mice.py b6-gcamp6s -u ```
*	``` python process-layer4-clusters-across-mice.py b6-gcamp6s -r ```

Data processing for S4a-c:
*	``` python process-layer4-clusters-across-mice-fraction.py -f 0```  
  Also run this separately for the three global shuffle controls, so adding the ```-sh```, ```-u``` and ```-r``` flags respectively (see Data processing for 1g-i)

Data processing for S5a-c:
* ``` python process-layer4-clusters-across-mice-fraction.py -f -0.3```  
  Also run this separately for the three global shuffle controls, so adding the ```-sh```, ```-u``` and ```-r``` flags respectively (Data processing for 1g-i)

Data processing for S5d-f:
* ``` python process-layer4-clusters-across-mice-fraction.py -f sign```  
  Also run this separately for the three global shuffle controls, so adding the ```-sh```, ```-u``` and ```-r``` flags respectively (Data processing for 1g-i)

Data processing for S6c-e:
*	``` python process-layer4-clusters-across-mice-caiman.py ```  
  Also run this separately for the three global shuffle controls, so adding the ```-sh```, ```-u``` and ```-r``` flags respectively (see Data processing for 1g-i)

Data processing for S7g-i:
*	``` python process-layer4-clusters-across-mice.py scn1a-jrgeco1a ```  
  Also run this separately for the three global shuffle controls, so adding the ```-sh```, ```-u``` and ```-r``` flags respectively (see Data processing for 1g-i)

#### Figure 1
* 1a, b: ``` python make-response-maps.py O03 2 -m 20 --examplecells 247 1 6 ```
* 1c-e: ``` python make-od-cluster-maps.py O03 ```
* 1f: ``` python make-retinotopy-scatter-maps.py O03 ```
* 1g-i: ``` python show-layer4-clusters-across-mice.py b6-gcamp6s ```

#### Supplementary Figure 1
*	S1a (row 1, M03): ``` python make-response-maps.py O06 1 -maxhls 20 ```
*	S1a (row 2, M01): ``` python make-response-maps.py O02 1 -maxhls 20 ```
*	S1a (row 3, M08): ``` python make-response-maps.py O12 1 -maxhls 30 ```
*	S1a (row 4, M05): ``` python make-response-maps.py O09 1 -maxhls 30 ```
*	S1b,c,e (row 1, M03): ``` python make-od-cluster-maps.py O06 --extendeddata ```
*	S1b,c,e (row 2, M01): ``` python make-od-cluster-maps.py O02 --extendeddata ```
*	S1b,c,e (row 3, M08): ``` python make-od-cluster-maps.py O12 --extendeddata ```
*	S1b,c,e (row 4, M05): ``` python make-od-cluster-maps.py O09 --extendeddata ```
*	S1d (row 1, M03): ``` python make-retinotopy-scatter-maps.py O06 ```
*	S1d (row 2, M01): ``` python make-retinotopy-scatter-maps.py O02 ```
*	S1d (row 3, M08): ``` python make-retinotopy-scatter-maps.py O12 ```
*	S1d (row 4, M05): ``` python make-retinotopy-scatter-maps.py O09 ```

#### Supplementary Figure 2
*	S2a, d, e-g: ``` python make-tiled-func-maps.py O10 ```
*	S2b, c: ``` python make-tiled-func-ret-maps.py O10 ```

#### Supplementary Figure 3
*	S3a-c: ``` python show-layer4-clusters-singletrials.py O03 ```
*	S3d-f: ``` python show-layer4-clusters-singletrials.py O09 ```
* S3g,h: ``` python show-layer4-clusters-across-mice-halftrials.py ```
* S3i: ``` python show-layer4-clusters-across-mice-stdev.py ```

#### Supplementary Figure 4
*	S4a-c: ``` python show-layer4-clusters-across-mice-fraction.py -f 0 ```
*	S4d: ``` python show-layer4-clusters-across-mice-odihistograms.py ```
*	S4e,f: ``` python show-layer4-clusters-across-mice-responseamplitudes.py ```
*	S4g: ``` python show-layer4-clusters-across-mice-ipsicontraseparate.py ```

#### Supplementary Figure 5
*	S5a-c: ``` python show-layer4-clusters-across-mice-fraction.py -f -0.3 ```
*	S5d-f: ``` python show-layer4-clusters-across-mice-fraction.py -f sign ```

#### Supplementary Figure 6
* S6a: ``` python make-od-cluster-maps-preprocesscompare.py suite2p ```
* S6b: ``` python make-od-cluster-maps-preprocesscompare.py caiman ```
* S6c-e: ``` python show-layer4-clusters-across-mice-caiman.py ```

#### Supplementary Figure 7
*	S7a,b: ``` python make-response-maps.py O40 1 -maxhls 30 -reverseodi --examplecells 8 10 7 ```
* S7c-e: ``` python make-od-cluster-maps.py O40 ```
* S7f: ``` python make-retinotopy-scatter-maps.py O40 ```
* S7g-i: ``` python show-layer4-clusters-across-mice.py scn1a-jrgeco1a ```


---

### Part 2: Column-like vertical organization for Ocular Dominance across cortical layers 2/3, 4 and 5

These paths are just FYI:  
Code path: “./code/part2-od-columns-layer2345”  
Data paths: “./data/part2-responsemaps-od-layer2345”, “./data/part2-planedata-od-layer2345”, “./data/part2-sideviewdata-od-layer2345”, “./data/part2-odimapsdata-od-layer2345”, “./data/part2-ccmapsdata-od-layer2345”, “./data/part2-roi-footprints”  
Processed data path: “./data/part2-processeddata"  

Note: There is already processed data supplied in the repository, however, in case you like to rerun it, the code below will recreate the processed data

Data processing for 2c-e:
*	``` python process-layer2345-columns-across-mice.py ```
*	``` python process-layer2345-columns-across-mice.py -sh ```
*	``` python process-layer2345-columns-across-mice.py -u ```
*	``` python process-layer2345-columns-across-mice.py -r ```

Data processing for S9:
*	``` python process-layer2345-sideviews-across-mice.py -s YZ ```  
*	``` python process-layer2345-sideviews-across-mice.py -s XZ ```  

Data processing for S10:
*	``` python process-layer2345-columns-across-mice.py -d 4 ```  
  Also run this separately for the three global shuffle controls, so adding the ```-sh```, ```-u``` and ```-r``` flags respectively (see Data processing for 2c-e)

Data processing for 2f-i and S11 (step 1):
*	``` python process-cellbased-odimaps-across-mice.py ```  
*	``` python process-cellbased-odimaps-across-mice.py -sh 0 ```  
*	``` python process-cellbased-odimaps-across-mice.py -sw 0 -d 100 ```  
*	``` python process-cellbased-odimaps-across-mice.py -sw 0 -d 200 ```  
  Run this 10 times for each of the shuffle controls (so with ```-sh 0```, ```-sh 1```, ..., ```-sh 10```, with ```-sw 0 -d 100```, ```-sw 1 -d 100```, ..., ```-sw 10 -d 100```, and ```-sw 0 -d 200```, ```-sw 1 -d 200```, ..., ```-sw 10 -d 200```. )

Data processing for 2f-i and S11 (step 2):
*	``` python process-cellbased-crosscorrelationmaps-across-mice.py ``` (real data)  
*	``` python process-cellbased-crosscorrelationmaps-across-mice.py  -sh 0 ``` (shuffled odi)
*	``` python process-cellbased-crosscorrelationmaps-across-mice.py  -pl 0 ``` (select plane from random other mouse)
*	``` python process-cellbased-crosscorrelationmaps-across-mice.py  -sw 0 -d 100``` (local swap of odi values at 100 micron distance)
*	``` python process-cellbased-crosscorrelationmaps-across-mice.py  -sw 0 -d 200``` (local swap of odi values at 200 micron distance)
  Run this 10 times for each of the shuffle controls, replacing the '0' with other integer numbers, see step 1.

#### Figure 2
* 2a: ``` python make-odi-volume-slices.py O03 ```
* 2b (left): ``` python make-volume-imagebased-odimaps.py O03 ```
* 2b (right): ``` python make-volume-cellbased-odimaps.py O03 ```
* 2c-e: ``` python show-layer2345-columns-across-mice.py ```
* 2f-i: ``` python show-cellbased-crosscorrelationmaps.py ```

#### Supplementary Figure 8
*	S8a: ``` python make-volume-imagebased-odimaps.py O02 -n 12 ```
* S8b: ``` python make-odi-volume-slices.py O02 ```
*	S8c,e,g,i: Same as S8a, but replace mouse name ```O02``` with ```O03```, ```O06```, ```O09``` or ```O10``` respectively

#### Supplementary Figure 9
* S9a: ``` python show-layer2345-sideviews-across-mice.py YZ ```  
* S9b: ``` python show-layer2345-sideviews-across-mice.py XZ ```  

#### Supplementary Figure 10
* S10: ``` python show-layer2345-columns-across-mice.py -d 4 ```

#### Supplementary Figure 11
*	S11a (left, center): ``` python make-volume-cellbased-odimaps.py O10 ```
*	S11a (right), S11b-e: ``` python show-cellbased-crosscorrelationmaps.py ```

#### Supplementary Figure 14
* S14: ``` python python show-roi-footprints-across-layers-2345.py ```


---

### Part 3: L4 ipsi-clusters show a non-random spatial arrangement, but no relationship with the retinotopic map

These paths are just FYI:  
Code path: “./code/part3-od-map-relationships”  
Data paths: “./data/part1-planedata-od-layer4”, “./data/part1-planedata-retinotopy-layer4”  
Processed data path: "./data/part3-processeddata-layer4"

Note: There is already processed data supplied in the repository, however, in case you like to rerun it, the code below will recreate the processed data

Data processing for 3c-i, S12d-h, and S13:
*	``` python process-od-map-relations.py ```
*	``` python process-od-map-relations.py -shodi 0```
*	``` python process-od-map-relations.py -shori 0```
  Run this 10 times for each of the shuffle controls (so with ```-shodi 0```, ```-shodi 1```, ..., ```-shodi 10``` and with ```-shori 0```, ```-shori 1```, ..., ```-shori 10``` )

#### Figure 3
* 3a, b: ``` python make-example-maps.py ```
* 3c-i: ``` python show-od-map-relations.py ```

#### Supplementary Figure 12
* S12a, b: ``` python make-example-maps.py ```
* S12c: ``` python make-od-maps-two-timepoints.py ```
* S12 d-h: ``` python show-od-map-relations.py ```

#### Supplementary Figure 13
* S13: ``` python show-od-map-relations.py ```



