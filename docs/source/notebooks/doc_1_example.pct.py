# %% [markdown]
"""
# This is an example notebook

This notebook is evaluated and inserted into the documentation.

"""

# %% [markdown]
"""
Here is an image.

![A great city](./images/SFO.jpeg)

"""

# %%
# imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Make a plot
# Load Dataset
df = sns.load_dataset('iris')

# Plot
plt.figure(figsize=(10,8), dpi= 80)
sns.pairplot(df, kind="reg", hue="species")
plt.show()

# %% nbsphinx="hidden"
# This is a hidden code cell
# We can use this to turn the examples into additional tests without
# polluting the documentation with imports and assertions.
# Please don't use hidden cells in a way that would break the example
# if someone executed the cells one after the other on their local
# machine.
class A:
    def one():
        return 1

    def two():
        return 2
