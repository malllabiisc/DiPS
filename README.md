## DiPS

<p align="center">
  <img align="center" src="https://github.com/malllabiisc/DiPS/blob/master/images/dips_model.png" alt="...">
</p>

*Overview of DiPS:
during decoding to generate k paraphrases. At each time step, a set of N sequences V<sup>t</sup>is used to determine k &lt; N sequences (X<sup>∗</sup>) via submodular maximization . The above figure illustrates the motivation behind each submodular component. Please see Section 4 in the paper for details.

### Dependencies

- compatible with python 3.6
- dependencies can be installed using `requirements.txt`

### Dataset
- Quora
- Twitter

### Training
```
python -m src.main -mode train -gpu 0 -use_attn -bidirectional
```

### Decoding
```
python -m src.main -mode decode -selec submod -run_name <run_name> -beam_width 10 -gpu 0
```

### Citation
Please cite the following paper if you find this work relevant to your application
```tex
```

For any clarification, comments, or suggestions please create an issue or contact [ashutosh@iisc.ac.in](http://ashutoshml.github.io) or [Satwik Bhattamishra](satwik55@gmail.com]