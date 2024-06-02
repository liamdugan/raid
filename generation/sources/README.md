# Data Repository Overview

## Table of Contents
- [Reddit](#reddit)
- [Code](#code)
- [Reviews](#reviews)
- [Abstracts](#abstracts)
- [News](#news)
- [Poetry](#poetry)
- [Czech](#czech)
- [German](#german)
- [Recipes](#recipes)
- [Wiki](#wiki)
- [Books](#books)

### Reddit
- **Description**: Dataset of Reddit posts, including titles, posts, and corresponding subreddits.
- **Source**: [Huggingface - Reddit Title Body](https://huggingface.co/datasets/sentence-transformers/reddit-title-body)
- **Entries**: 2000
- **Edits**:
  - Selected only 2021 data.
  - Dropped texts outside of 700-1600 chars.
  - Removed texts with special symbols.

### Code
- **Description**: Code exercises paired with correct solutions in Python.
- **Source**: Austin, Jacob, et al. ["Program synthesis with large language models"](https://arxiv.org/abs/2108.07732) (2021).
- **Entries**: 974
- **Edits**: Removed duplicates.

### Reviews
- **Description**: Dataset of movies and their corresponding IMDb reviews.
- **Source**: Aditya Pal, et al. ["IMDb Movie Reviews Dataset"](https://dx.doi.org/10.21227/zm1y-b270) (2020).
- **Entries**: 1150

### Abstracts
- **Description**: Dataset containing titles of ArXiv papers published in 2023 alongside their abstracts.
- **Source**: [Kaggle - Arxiv Dataset by Cornell University](https://www.kaggle.com/datasets/Cornell-University/arxiv)
- **Entries**: 2000
- **Edits**:
  - Selected only 2023 entries.
  - Dropped texts outside of 700-1600 chars.

### News
- **Description**: Dataset of BBC articles and their titles.
- **Source**: D. Greene and P. Cunningham. ["Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering"](https://dl.acm.org/doi/10.1145/1143844.1143892), ICML 2006.
- **Entries**: 2000
- **Edits**: Combined and sampled articles from various topics.

### Poetry
- **Description**: Dataset of poems spread across genres, each with a title.
- **Source**: [Kaggle - Poems Dataset](https://www.kaggle.com/datasets/michaelarman/poemsdataset)
- **Entries**: 2000
- **Edits**: Dropped texts outside of 700-1600 chars.

### Czech
- **Description**: Dataset of articles written in Czech along with their titles.
- **Source**: Boháček, et al. [“Czech-ing the News: Article Trustworthiness Dataset for Czech.”](https://aclanthology.org/2023.wassa-1.10/) WASSA 2023.
- **Entries**: 2000
- **Edits**: Dropped texts outside of 700-1600 chars.

### German
- **Description**: Dataset of articles written in German along with their titles.
- **Source**: Schabus, Dietmar, et al. ["One Million Posts: A Data Set of German Online Discussions"](https://dl.acm.org/doi/10.1145/3077136.3080711) (2017).
- **Entries**: 2000
- **Edits**: Dropped texts outside of 700-1600 chars.

### Recipes
- **Description**: Dataset containing recipes, each with a title, list of ingredients, and directions.
- **Source**: Bień, et al. ["RecipeNLG: A Cooking Recipes Dataset for Semi-Structured Text Generation"](https://aclanthology.org/2020.inlg-1.4.pdf), INLG 2020.
- **Entries**: 2000
- **Edits**: Dropped texts outside of 700-1600 chars.

### Wiki
- **Description**: Dataset of Wikipedia page titles paired with their introductory texts.
- **Source**: [Huggingface - GPT Wiki Intro](https://huggingface.co/datasets/aadityaubhat/GPT-wiki-intro)
- **Entries**: 2000
- **Edits**: Dropped texts outside of 700-1600 chars.

### Books
- **Description**: Dataset containing book titles and summaries of their plots.
- **Source**: Bamman, David & Smith, Noah. ["New Alignment Methods for Discriminative Book Summarization"](https://arxiv.org/abs/1305.1319) (2013).
- **Entries**: 2000
- **Edits**: Dropped texts outside of 700-1600 chars.
