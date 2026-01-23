# Dual Channel GTCRN.
A training code template is highly valuable for deep learning engineers as it can significantly enhance their work efficiency. Despite different programmers have varying coding styles, some are excellent while others may not be as good. My philosophy is to prioritize simplicity. In this context, I am sharing a practical organizational structure for training code files in speech enhancement (SE). The primary focus is on keeping it concise and intuitive rather than aiming for comprehensiveness.

## Usage
Download and extract  
[LibriSpeech ASR corpus](https://openslr.trmal.net/resources/12/train-clean-360.tar.gz)  
[MUSAN](https://openslr.trmal.net/resources/17/musan.tar.gz)  
[noise](https://drive.google.com/file/d/1tY6qkLSTz3cdOnYRuBxwIM5vj-w4yTuH/view?usp=drive_link)

`python prep_data.py`  

`python train.py`  

`python export.py`  

## Note  

Also to add TTS data of command sentences of Wakeword of use and simple random commands such as 'Octavia turn on the light, Octavia play some music, Octavia...'
To be implemented.  
`python train-resume.py`  added full session save incase of training failure so you can resume and better validation logic  


## Acknowledgement
This code is basically all the great work by [Rong Xiaobin](https://github.com/Xiaobin-Rong)  
[GTCRN](https://github.com/Xiaobin-Rong/gtcrn)  
https://github.com/Xiaobin-Rong/SEtrain  
https://github.com/Xiaobin-Rong/TRT-SE




