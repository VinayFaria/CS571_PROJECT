import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy

# packing audio signal samples in frames
def enframe(x, winsize, hoplength, fs):
    # compute frame length and frame step (convert from seconds to samples)
    winsize = int(math.ceil(winsize * fs))
    hop_size = int(math.ceil(hoplength * fs))
    signal_length = len(x)
    frames_overlap = winsize - hop_size
    
    num_frames = np.abs(signal_length - frames_overlap) // np.abs(winsize - frames_overlap)
    rest_samples = np.abs(signal_length - frames_overlap) % np.abs(winsize - frames_overlap)
    
    # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal
    if rest_samples != 0:
        pad_signal_length = int(hop_size - rest_samples) # Dividend = Divisor × Quotient + Remainder
        z = np.zeros((pad_signal_length))
        pad_signal = np.append(x, z)
        num_frames += 1
    else:
        pad_signal = x
    
    num_frames = int(num_frames)
    # making index for each sample in each frame row contain particular frame particular frame contain number of samples i.e. frame length
    idex1 = np.tile(np.arange(0, winsize), (num_frames, 1))
    idex2 = np.tile(np.arange(0, num_frames * hop_size, hop_size),(winsize, 1)).T
    indices = idex1 + idex2
    frames = pad_signal[indices.astype(np.int64)]#, copy=False)]
    
    rect_frames = frames.copy()
    hamming_frames = frames.copy()
    
    # modifying frames for hamming
    for i in range(len(hamming_frames)):
        dummy = np.hamming(winsize)
        hamming_frames[i] = hamming_frames[i]*dummy
    
    # checking window type and then modifying frames accordingly
    j = 0
    for frame in frames:
        dummy = np.hamming(winsize)
        hamming_frames[j] = hamming_frames[j]*dummy
        j += 1
    """
    for frame in frames:
        if wintype == 'rect':
            pass
        elif wintype == 'hamm':
            dummy = np.hamming(winsize)
            frames[j] = frames[j]*dummy
        j += 1
    """
    return rect_frames,hamming_frames, num_frames

# Load data from wav file
#Default Setting - sub-sampling to default 22,050 Hz, Explicitly Setting sr=None ensures original sampling preserved
y, srl = librosa.load('should_we_chase.wav',sr=None)
print('duration of audio is:',librosa.get_duration(y=y, sr=srl))
print('sampling rate of audio is:',srl)
window_size = 0.03
hop_length = 0.03
