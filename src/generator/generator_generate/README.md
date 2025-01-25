# generator generate

Prepares data and generates paraphrases.
Given a set of human phrases, the generator generates machine paraphrases
of each human phrase it receives.

- `generator_generate` prepares data and performs the generation.

```
GENERATOR GENERATING

     GENERATOR --->  paraphrases_generated.tsv
       ^
       |
phrases_input.txt
    (predict)
```

## Directory structure

Directory structure
```
data
|-input
  |-phrases_input.txt
|-output
  |-paraphrases_generated.tsv
