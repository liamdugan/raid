# RAID Dataset Generation

## Generation
Dataset generation for RAID is performed by the following scripts:
- `sample.py`: Sample human-written sources to create the initial human dataframe
- `initialize.py`: Initialize the base dataframe with the set of generators to be included
- `generate.py`: Generate output for all empty rows in the input dataframe
- `validate.py`: Ensure that the dataframe passes all sanity checks
- `compute_metric.py`: Compute metrics such as perplexity, selfbleu, and token length

## Example
To generate a dataset of only ChatGPT and GPT-4 with one human-written source document per domain you would run the following scripts
```
$ python sample.py --num_samples 1 --domains all --output_path out.csv
...
$ python initialize.py --df_path out.csv --models chatgpt gpt4 --output_path init.csv
...
$ python generate.py --input init.csv --models all --output_path data.csv
...
$ python validate.py --data data.csv --check_null --check_null_generations ...
...
```

To compute a metric for the dataset (say, token lengths) you would run the following
```
$ python compute_metric.py --metric_name tokens --data_path data.csv --output_fname tokens.json
```


## Column Values
The RAID dataset has the following columns

1. `id`: A uuid4 that uniquely identifies the original content of the generation
2. `adv_source_id`: uuid4 that indicates the source of the generation if adversarial
3. `source_id`: uuid4 of the human-written source text
4. `model`: The model that generated the text
   - Choices: `['chatgpt', 'gpt4', 'gpt3', 'gpt2', 'llama-chat', 'mistral', 'mistral-chat', 'mpt', 'mpt-chat', 'cohere', 'cohere-chat']`
5. `decoding`: The decoding strategy used 
    - Choices: `['greedy', 'sampling']`
6. `repetition_penalty`: Whether or not we use a repetition penalty of 1.2 when generating
    - Choices: `['yes', 'no']`
7. `attack`: The adversarial attack used
    - Choices: `['homoglyph', 'number', 'article_deletion', 'insert_paragraphs', 'perplexity_misspelling', 'upper_lower', 'whitespace', 'zero_width_space', 'synonym', 'paraphrase', 'alternative_spelling']`
8. domain: The genre from where the prompt/text was taken
    - Choices: `['abstracts', 'books', 'code', 'czech', 'german', 'news', 'poetry', 'recipes', 'reddit', 'reviews', 'wiki']`
9. `title`: The title of the article used in the prompt
10. `prompt`: The prompt used to generate the text
11. `generation`: The text of the generation
12. `num_edits`: The number of adversarial edits made
13. `edits`: The spans of indices in the generation where the edits were made