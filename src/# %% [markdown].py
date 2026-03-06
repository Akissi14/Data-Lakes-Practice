# %% [markdown]
# # PFAM PROTEIN CLASSIFICATION

# %% [markdown]
# This notebook explores data in the PFAM Dataset, where each entry corresponds to a protein accompanied with additional information such as it's amino-acid sequence.
# 
# The final goals of this notebook are:
# * To gather descriptive statistics on amino-acid sequences in the dataset
# * Visualize key elements of the data
# * Infer a good way to approach the problem

# %% [markdown]
# # IMPORTS

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
data = pd.read_csv('data/bronze/combined_data.csv')

# %% [markdown]
# Let's verify that the data has been correctly loaded:

# %%
data.head()

# %% [markdown]
# Now that the data is loaded, we shall try to explore it to see **whether it is suitable for learning**. If not, **why it isn't, and how to transform it** to make it suitable for learning.
# 
# Some questions that first come to mind are the following:
# 
# 1.   Are classes well balanced? If not, how imbalanced are they? Are there imbalances from one data to another?
# 2.   How to efficiently compensate for these imbalances?
# 3.   How long are protein sequences on average ? In a real life scenarion, this would be a very important question to manage your GPU VRAM during training
# 
# 
# 
# 
# 
# 
# 
# 

# %% [markdown]
# # Class balance

# %% [markdown]
# As is, the data is the raw and transformations need to be applied to check for potential class imbalance. The PFAM dataset reportedly has entries from 17,930 output classes, therefore it seems likely that some will be over-represented and some under-represented.
# 
# This needs to be evaluated to develop a proper training protocol later on.

# %%
def munge_data(data: pd.DataFrame):
  '''
  Munges data to extract key insights

  Paramaters:
  data (pd.DataFrame): Raw DataFrame after unpacking

  Returns:
  (pd.DataFrame): Munged data 
  '''
  munged_data = {
      'sample_count': data.shape[0],
      'class_count': data['family_accession'].nunique(),
      'min_samples_per_class': data.groupby('family_accession').size().min(),
      'max_samples_per_class': data.groupby('family_accession').size().max(),
      'mean_samples_per_class': data.groupby('family_accession').size().mean(),
      'min_seq_length': data['sequence'].apply(len).min(),
      'max_seq_length': data['sequence'].apply(len).max(),
      'mean_seq_length': data['sequence'].apply(len).median()
  }

  return pd.DataFrame([munged_data])

# %%
data_munged = munge_data(data)
data_munged

# %% [markdown]
# The above analysis confirms the following insights:
# 
# * **Output classes seem to be extremely imbalanced** when looking at min, max and mean number of samples per class
# * **Sequence lengths look strongly unbalanced too**. However, with the median sequence length being 119, this tends to indicate that all context in most of the sequences can be captured with a model input_size of at least 119. In training, the goal will be to maximize this value to capture as much context as possible while staying in the range of available compute ressources.
# 
# 

# %% [markdown]
# Another point we want to verify is that amino-acid sequences are similarly distributed in terms of average length per class and samples per class between partitions.
# 
# Using seaborn, let's visualize these distributions to gain visual insight into the data:

# %% [markdown]
# # Class Distribution Plot

# %%
# Visualize protein count per family size to visualize class imbalance

fig, ax = plt.subplots(figsize=(20,10))
colors = ['blue']
sns.set_theme(style="whitegrid")

family_size = data['family_id'].value_counts()
class_distribution = family_size.value_counts().sort_index()

sns.histplot(data=class_distribution, bins=100, ax=ax, color=colors, kde=False, alpha=0.15)
plt.yscale('log')

fig.suptitle('PFAM Class Distribution')

# %%
class_distribution

# %%
# Visualize the distribution of sequence lenghts
fig, ax = plt.subplots(figsize=(20,10))
colors = ['green']

seq_lenghts = data['sequence'].apply(len).value_counts()

ax.set_xlabel('Sequence length')
ax.set_ylabel('Amino Acid count')

sns.histplot(data=seq_lenghts, bins=50, ax=ax, color=colors, kde=True, alpha=0.15)

ax.set_title('Distribution of amino acids per sequence')

# %% [markdown]
# Some amino acid sequences are extremely long. However, those are a minority, as both the graph and earlier statistics show that AA sequence length tends to gravitate towards the lower end of the spectrum.

# %% [markdown]
# # Compensate the imbalance
# 
# 

# %% [markdown]
# As we have seen, a large amount of classes only have 2 or 1 representatives for 3 sets. To split the data into a train, val and test set, you will need to think of a custom way to stratify the split accross these three sets. *Imagine you are a Data Engineer in a real life scenario, and the priority is to have data to evaluate the performance of the model the Machine Learning Engineers you work with will train*.
# 
# 


