# Project 13. Personality Analysis

This project explores the link between emotion and personality in dialogue systems. We especially focus on the five-trait personality model (extraversion, agreeableness, openness, conscientiousness, neuroticism). For this purpose, we explore the **Persona-E2** dataset (see the Persona-E2 Project Page), a large-scale human-grounded dataset mapping personality traits alongside emotional responses and appraisal shifts across news, social media and life narratives. You may study the implementation provided by the authors that uses an LLM-based approach as well.

---

## Tasks

### 1. General trend of the conversation data

We want to explore the general trend of the conversation data. For this purpose, create a dataframe for each speaker by gathering all his utterances in a single file. Then write a script that determines, for each speaker:

- the vocabulary set
- the vocabulary size
- the total number of tokens
- the total number of repetitions of words in the same post
- the total number of confirmation words (e.g., *yes*, *OK*, *sure*)
- the total number of negation tokens

Summarize the result in a table, and then draw a subgraph that shows on the same plot the evolution of the number of repetitions, number of negations, and number of confirmation-like tokens, with respect to the number of tokens employed for each speaker. (You may create some subdivision from the total number of tokens to ensure enough data points are used to represent the graphical illustration.)

Calculate the overall personality for each speaker by averaging over all instances of the original dataset, and comment on possible similarities and differences between speakers and whether some attributes are more associated with some personality patterns.

### 2. Topic change during conversation

We want to evaluate the extent to which a speaker changes topic during conversation. In an attempt to quantify this generic trend, we hypothesize this is the case if the speaker quickly introduces new vocabulary and yields a higher number of topics. To test this hypothesis:

- **(a)** Suggest a script that plots the evolution of the vocabulary size with respect to the number of tokens employed (draw on the same plot the distribution of each speaker), then comment on the suitability of the approach to discriminate key personality traits.
- **(b)** To test the second aspect, suggest a script that uses LDA to determine the optimal number of topics using global measures such as coherence and perplexity. You may draw inspiration from tutorials such as *Demystifying Topic Modeling Techniques in NLP* by Vijay Choubey (Medium). Similarly, comment on the suitability of the approach to highlight some personality traits.

### 3. Emotion patterns and personality traits

We want to test whether some emotion patterns occur with some particular personality trait. For this purpose, we use the **NRC VAD lexicon**, where around 20K words have been assigned values for the valence, arousal and dominance (VAD) dimensions (see `EmotionDynamics/lexicons` at master, Priya22/EmotionDynamics on GitHub).

Write a script that identifies NRC lexicon terms from the dataframe of each speaker and calculates:

- the average valence, arousal and dominance score, normalized with respect to the total number of lexicon terms in the dataframe
- the corresponding standard deviations
- the minimum and maximum values

Comment on whether some valence, arousal, dominance values provide insights to discriminate some personality traits.

### 4. Individual personality-trait cases

Now, instead of considering overall speaker utterances, we want to investigate individual personality trait cases. For this purpose:

1. Write a script that reports the **dominant personality trait** from the scores of the five personality traits.
2. Construct a new dataframe for each personality trait by gathering all utterances that fall in this category.
3. Repeat the process of estimating:
   - the vocabulary set
   - the vocabulary size
   - the number of repetitions in the same post
   - the number of confirmation-like tokens
   - the number of negation-like tokens
4. For each dataframe, draw a plot showing the evolution of the size of vocabulary, number of negation-like tokens, and number of confirmation-like tokens, with respect to the total number of tokens employed (you may create a subdivision from the total number of tokens of each dataframe).
5. Comment on whether some specific discrepancy can be spotted among the various dataframes.

### 5. Embedding-based similarity to trait definitions

We want to consider the definitions of each personality trait given in the figure below (as printed in Larsen and Buss, 2009):

| Trait | Definition terms |
|---|---|
| **I. Extraversion or Surgency** | Talkative–silent<br>Sociable–reclusive<br>Adventurous–cautious<br>Open–secretive |
| **II. Agreeableness** | Good-natured–irritable<br>Cooperative–negativistic<br>Mild/gentle–headstrong<br>Not jealous–jealous |
| **III. Conscientiousness** | Responsible–undependable<br>Scrupulous–unscrupulous<br>Persevering–quitting<br>Fussy/tidy–careless |
| **IV. Emotional stability** | Calm–anxious<br>Composed–excitable<br>Not hypochondriacal–hypochondriacal<br>Poised–nervous/tense |
| **V. Culture—Intellect, Openness** | Intellectual–unreflective/narrow<br>Artistic–nonartistic<br>Imaginative–simple/direct<br>Polished/refined–crude/boorish |

Suggest a script that uses an approach for generating an embedding of each dataframe and another embedding for each definition term, then uses cosine similarity to evaluate the similarity between the dataframe and the personality definition. In particular, if personality trait A has multiple definitions, say T1, T2, T3, then for dataframe embedding T, the score assigned to trait A is computed as:

```
Score(A) = max( cosine(embedding(T1), T), cosine(embedding(T2), T), cosine(embedding(T3), T) )
```

You may use doc2vec embedding or DistilBERT embedding.

Comment on the compatibility between the embedding-based personality and the labelled dataset.

### 6. Explanation via LLM prompting

We want to provide an explanation for the findings. For this purpose, suggest an LLM prompting strategy that would allow you to ask the model why a specific personality trait has been generated. Demonstrate your findings and provide examples of visualization for various personality traits.

### 7. Advanced implementation

Now we want to study the advanced implementation reported in the provided database link. Use your own initiative to seek to integrate explanation of the findings and use the same performance metrics employed by the authors to demonstrate any improvement in performance.

### 8. Literature, limitations and discussion

Use appropriate literature to comment on the findings. Also, identify any additional input that would allow you to further elucidate any of the preceding, and use appropriate corpus linguistics literature to justify your findings and comment on the obtained results. Finally, comment on the limitations and structural weaknesses of the data processing pipeline.
