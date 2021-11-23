# CS571_PROJECT
# Formant estimation from speech signals

## Summary
When the speech signals are analysed in frequency domain i.e. spectrum analysis then we get to know about frequencies which are present. The resonances that occur at certain frequencies are nothing but formants. If the full speech signal is being analysed at once then multiple formants will be seen, but we can't figure it out when a particular formant occured. Therefore we analyse the signal in small chunks using STFT (Short Time Fourier Transform) by appropriately defining window_length (Window length is the length of the fixed intervals in which STFT divides the signal) and hop_length (Hop length is the length of the non-intersecting portion of window length). In this project for every frame, formants were identified.

## Useful links
- [Digital Speech Processing Course by Lawrence Rabiner](https://web.ece.ucsb.edu/Faculty/Rabiner/ece259/)

### Final Code
Python code can be found here [code.py](https://github.com/VinayFaria/CS571_PROJECT/blob/main/code.py)

### Plots and audio files
- Plots are available here 
- Audio files are available here 
