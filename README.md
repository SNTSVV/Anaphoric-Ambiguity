# Anaphoric-Ambiguity

This repo contains the replication package for the ICSE 22 technical paper, titled "[Automated Handling of Anaphoric Ambiguity in Requirements: A Multi-solution Study](https://dl.acm.org/doi/10.1145/3510003.3510157)" 

## Folder Structure: 
We organize the *SupplementaryMaterial* folder into three different sub-folders: 
1. **Datasets**: This sub-folder contains the three datasets that are used in our study. These are: (1) the *DAMIR* dataset which we constructed as part of our work, and two publicly available datasets which we adapted to our work (2) *CoNLL2011*, and (3) *ReqEval*. More details on these datasets are provided in Section 4.3 of the paper.

2. **Application**: This sub-folder contains the final solutions resulting from our study, and the Jupyter notebook containing the scripts that one can apply to run our solutions. We will explain in detail how to run this notebook.
We note that we did not provide a runanble script for the last solution (solution 6 in our paper) that is based on the existing NLP coreference resolution tools. This is because this solution is not compatible with the other Python libraries that we use for the other solutions. Given that solution 6 did not achieve good-enough results compared with the SpanBERT- and ML-based solutions, we decided to drop solution 6 from our supplementary material to simplify the application environemnt for the other five solutions.      

3. **Tuning**: This sub-folder contains the notebook required for fine-tuning the SpanBERT-based solutions and experimenting with different configurations for the ML-based solutions. More details are provided in Section 4.5 of our paper. 

## How to apply the SpanBERT- and ML-based solutions for anaphoric ambiguity handling in a simple example? 

### (A) Download the **Application** folder.

1. Clone the repo and navigate to the **Application** folder

```
git clone https://github.com/SNTSVV/Anaphoric-Ambiguity.git
cd Anaphoric-Ambiguity/Application
```

2. Download SpanBERT_NLP and SpanBERT_RE model weights separatly from [this google drive link](https://drive.google.com/file/d/1uMZpbIMv-bP_T0x-E2FYyT8ECrF5f3z8/view?usp=share_link).

3. Unzip the downloaded folder and move the content to the **Application** folder as follows:
```
Application/SpanBERT-NLPv21.8.10/
Application/SpanBERT-REv21.9.01/
```

Alternatively from the command line (make sure you are in the Application folder):
```
pip install gdown 
gdown https://drive.google.com/uc?id=1uMZpbIMv-bP_T0x-E2FYyT8ECrF5f3z8
unzip anaphoric-ambiguity-models.zip
rm anaphoric-ambiguity-models.zip
```


### (B) Install Python 3, pip, and Jupyter Notebook.
1. **Python 3.8 or higher**: Get the latest Python version [here](https://www.python.org/downloads/). If you already have Python installed, make sure that the version you have is 3.8 or later. You can check the Python version installed on your computer by typing this command in the terminal window: 
```
$ python --version
```
2. **pip**: Make sure you have *pip* (the package installer for Python) installed. More information can be found [here](https://pip.pypa.io/en/stable/installation/).  
3. **Jupyter Notebook**: Install Jupyter notebook. The installation instructions are provided in [the Jupyter documentation](https://test-jupyter.readthedocs.io/en/latest/install.html). It is highly recommended that you install Jupyter using *Anacoda*. 

### (C) Install the required Python libraries.   
We have created a list *requirements.txt* within the **Application** folder that includes the Python libraries we use in our work. You can download all required libraries by typing the following command in a terminal: 
```
$ pip install -r requirements.txt
```

### (D) Run the solutions 
1. Launch the Jupyter Notebook application using the following command: 
```
$ Jupyter notebook
```
2. The Jupyter Notebook will be launched in a new browser window (or a new tab) showing the Notebook Dashboard. Navigate through the folder structure on the first page of Jupyter (*Files* tab) and go to the folder where you have downloaded the **Application**.   
3. Open the Jupyter notebook **Solutions application.ipynb**.
4. Select the option *Cell>Run All* to execute all of the cells in the notebook. 


