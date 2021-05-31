House-GAN++
======

Code and instructions for our paper:
[House-GAN++: Generative Adversarial Layout Refinement Network towards Intelligent Computational Agent for Professional Architects](https://arxiv.org/abs/2103.02574), CVPR 2021. Project [website](https://ennauata.github.io/houseganpp/page.html).

Data
------
![alt text](https://github.com/ennauata/houseganpp/blob/main/refs/sample.png "Sample")
We have used the [RPLAN dataset](http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/index.html), which offers 60k vector-graphics floorplans designed by professional architects. Qualitative and quantitative evaluations based on the three standard metrics (i.e., realism, diversity, and compatibility) in the literature demonstrate that the proposed system outperforms the current-state-of-the-art by a large margin.<br/>
<br/>

Demo
------
![image](https://user-images.githubusercontent.com/719481/116904118-29674080-abf2-11eb-8789-62c36edc4f9b.png)
Please check out our live [demo](http://www.houseganpp.com).

Running pretrained models
------
***See requirements.txt for checking the dependencies before running the code***

For running a pretrained model check out the following steps:
- Run ***python test.py***.
- Check out the results in output folder.

Training models
------
Coming Soon!

Citation
------
```
@misc{nauata2021housegan,
      title={House-GAN++: Generative Adversarial Layout Refinement Networks}, 
      author={Nelson Nauata and Sepidehsadat Hosseini and Kai-Hung Chang and Hang Chu and Chin-Yi Cheng and Yasutaka Furukawa},
      year={2021},
      eprint={2103.02574},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

Contact
------
If you have any question, feel free to contact me at nnauata@sfu.ca


Acknowledgement
------
This research is partially supported by NSERC Discovery Grants, NSERC Discovery Grants Accelerator Supplements, DND/NSERC Discovery Grant Supplement, and Autodesk. We would like to thank architects and students for participating in our user study.
