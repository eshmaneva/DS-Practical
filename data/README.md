This folder contains the data files used in the model.

The dataset used as the input for the **T5 Summarization model** is the Amazon Product Review dataset. It is a publicly available dataset, which contains
568.454 reviews. The data is stored within 10 columns: Id, ProductId, UserId, ProfileName, HelpfulnessNumerator, HelpfulnessDenominator, Score, Time, Summary, and Text.

**Classification model** uses the same dataset as an initial input for the training. However, after training the csv-file with T5-generated summaries is used.
