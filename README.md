# CS571_PROJECT
# Formant estimation from speech signals

## Summary
When the speech signals are analysed in frequency domain i.e. spectrum analysis then we get to know about frequencies which are present. The resonances that occur at certain frequencies are nothing but formants. If the full speech signal is being analysed at once then multiple formants will be seen, but we can't figure it out when a particular formant occured. Therefore we analyse the signal in small chunks using STFT (Short Time Fourier Transform) by appropriately defining window_length (Window length is the length of the fixed intervals in which STFT divides the signal) and hop_length (Hop length is the length of the non-intersecting portion of window length). In this project for every frame, formants were identified.

## Useful links
- [Digital Speech Processing Course by Lawrence Rabiner](https://web.ece.ucsb.edu/Faculty/Rabiner/ece259/)
- [Relation between sampling rate and frequency range](http://www.asel.udel.edu/speech/tutorials/instrument/sam_rat.html)
- [Linguistic Phonetics Spectral Analysis](https://ocw.mit.edu/courses/linguistics-and-philosophy/24-915-linguistic-phonetics-fall-2015/lecture-notes/MIT24_915F15_lec6.pdf)
- [Signal framing](https://superkogito.github.io/blog/SignalFraming.html)

## Final Code
Python code can be found here [code.py](https://github.com/VinayFaria/CS571_PROJECT/blob/main/code.py)

## Plots and audio files
- Audio files are available here [Audio Files](https://github.com/VinayFaria/CS571_PROJECT/tree/main/Audio%20Files)
- Plots are available here [Plots and Formants text file](https://github.com/VinayFaria/CS571_PROJECT/tree/main/Plots)

## Limitation
- Not work for Ill-Conditioning. Theory for Ill-Conditioning can be found [here](http://www-mmsp.ece.mcgill.ca/Documents/Reports/2003/KabalR2003a.pdf)
- audio files s2, s3, s4, s5 and s6 are example of Ill-Conditioning for window_size = 0.049sec and hop_length = 0.01sec
