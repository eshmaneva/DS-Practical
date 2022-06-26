## LMU DS Practical project in collaboration with BMW

When (or after) the car is produced, different defects occur. These defects are recorded and stored in the data source – the **“Knowledge base”** – that summarizes similar defects and assigns them to the prebuilt defect cluster. Each defect contains high amount of human written-text data, which makes analysis time-consuming and complicated. 

Hence, to ease the inspection of such data, the **goal** of this project was set to *building a model that will process the human created text data of different length, create a summary of it, classify it based on the „sense“ of the generated summary and evaluate the quality of it*. Shortly speaking, a model, that reads human written text, creates a summary for it and classifies it to one of the pre-defined classes.


### The structure of the repository:
* **Data folder** \
Contains README file with the description of the data and the link to the dataset and a csv file with the examples of the generated summaries
* **Documentation folder** \
Contains the final report, that presents the whole information about the project, and the corresponding slides
* **Model folder** \
Contains python files for both classification and summarization models in two formats each: a standart *'\*\.py'* python file and the *'\*\.ipynb'* notebook document
* **Supplementary folder** \
Contains the code for the first-choice model for summarization, that was later abandoned (c.f. DS-Practical/documentation/Report_Data Quality Metrics.pdf --> Ch. 3.4 Model Choice) in both formats: *'\*\.py'* and *'\*\.ipynb'*
