#=============================================================================
# Loading library
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy

#=============================================================================
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
    
    return rect_frames,hamming_frames, num_frames

#=============================================================================
# Load data from wav file
# Default Setting - sub-sampling to default 22,050 Hz, Explicitly Setting sr=None ensures original sampling preserved
name_read = "s6"
filename_read = "%s.wav" % name_read
y, srl = librosa.load(filename_read,sr=10000)

name_write = name_read + "_formants"
filename_write = "%s.txt" % name_write
file1 = open(filename_write,"w")
file1.write("duration of audio: %s seconds \n" %librosa.get_duration(y=y, sr=srl))
file1.write("Sampling rate of audio: %s \n" %srl)
#print('duration of audio is:',librosa.get_duration(y=y, sr=srl))
#print('sampling rate of audio is:',srl)
window_size = 0.049 #enter in seconds
hop_length = 0.01 #enter in seconds
file1.write("window_size: %s \n" %window_size)
file1.write("hop_length: %s \n" %hop_length)
file1.write("\n")
file1.write("formants are \n")

#=============================================================================
# Plot sound wave
plt.figure(1)
plt.plot(np.linspace(0, librosa.get_duration(y=y, sr=srl), num=len(y)), y)
plt.xlabel('Time [seconds]')
plt.ylabel('Amplitude')
plt.title('Sound wave in time domain')
plt.grid()

#=============================================================================
# Calling Enframe function
rect_window_frames,hamming_window_frames,num_frames = enframe(y, window_size, hop_length, srl)

#=============================================================================
# impulse generator
n=0
impulse = []
x_axis = np.arange(0, len(hamming_window_frames[0]), 1)
for i in x_axis:
    if i==n:
        impulse.append(1)
    else:
        impulse.append(0)

#=============================================================================
# saving formants in text file
j = 0
for frame in hamming_window_frames:
    p = int(srl/1000)+2 # number of poles
    #A1 = librosa.lpc(single_frame_rect , p)
    A2 = librosa.lpc(frame, p)
    
    #inverse_filter_rect = scipy.signal.lfilter([1], A1, impulse)
    inverse_filter_hamm = scipy.signal.lfilter([1], A2, impulse)
    
    #fft_single_frame_rect = np.log10(abs(np.fft.fft(single_frame_rect)))
    fft_single_frame_hamm = np.log10(abs(np.fft.fft(frame)))
    #fft_inverse_filter_rect = np.log10(abs(np.fft.fft(inverse_filter_rect)))
    fft_inverse_filter_hamm = np.log10(abs(np.fft.fft(inverse_filter_hamm)))
    
    
    fft_fre = np.fft.fftfreq(n=frame.size, d=1/srl)
    fft_fre = fft_fre[0:len(fft_fre)//2]
    #print(fft_fre)
    
    samples_in_frame = len(fft_inverse_filter_hamm)
    #fft_single_frame_rect = fft_single_frame_rect[0:samples_in_frame//2]
    fft_single_frame_hamm = fft_single_frame_hamm[0:samples_in_frame//2]
    #fft_inverse_filter_rect = fft_inverse_filter_rect[0:samples_in_frame//2]
    fft_inverse_filter_hamm = fft_inverse_filter_hamm[0:samples_in_frame//2]
    
    peak_location_hamm = scipy.signal.find_peaks(fft_inverse_filter_hamm)
    
    peak_amplitude_hamm = []
    #print('The formants in frame when hamming window is considered are: ')
    for i in peak_location_hamm[0]:
        peak_amplitude_hamm.append(fft_inverse_filter_hamm[i])
    peak_amplitude_hamm = np.asarray(peak_amplitude_hamm)
    peak_location_hamm = peak_location_hamm[0]/(samples_in_frame//2)
    peak_location_hamm = peak_location_hamm*(srl//2)
    peak_location_hamm = peak_location_hamm.tolist()
    
    file1.write("frame %s:" %j)
    for i in range(len(peak_location_hamm)):
        if i == len(peak_location_hamm)-1:
            peak_location_hamm[i] = str(peak_location_hamm[i]) + '\n'
        else:
            peak_location_hamm[i] = str(peak_location_hamm[i]) + ' '
    file1.writelines(peak_location_hamm)
    
    j +=1
    
file1.close() #to change file access modes

#=============================================================================
# Printing formants based on user input frame number
while True:
    frame_number = input('Enter any frame number from 0 to {} to get its magnitude spectrum or press enter to exit: '.format(num_frames-1))
    if not frame_number or int(frame_number)> num_frames:
        break
    
    #single_frame_rect = rect_window_frames[int(frame_number)]
    single_frame_hamm = hamming_window_frames[int(frame_number)]
    
    p = int(srl/1000)+2 # number of poles
    #A1 = librosa.lpc(single_frame_rect , p)
    A2 = librosa.lpc(single_frame_hamm, p)
    
    #inverse_filter_rect = scipy.signal.lfilter([1], A1, impulse)
    inverse_filter_hamm = scipy.signal.lfilter([1], A2, impulse)
    
    #w1, h1 = scipy.signal.freqz(1,A1)
    w2, h2 = scipy.signal.freqz(1,A2)
    
    #fft_single_frame_rect = np.log10(abs(np.fft.fft(single_frame_rect)))
    fft_single_frame_hamm = np.log10(abs(np.fft.fft(single_frame_hamm)))
    #fft_inverse_filter_rect = np.log10(abs(np.fft.fft(inverse_filter_rect)))
    fft_inverse_filter_hamm = np.log10(abs(np.fft.fft(inverse_filter_hamm)))
    #fft_h1 = np.log10(np.abs(h1))
    #fft_h2 = np.log10(np.abs(h2))
    
    
    fft_fre = np.fft.fftfreq(n=single_frame_hamm.size, d=1/srl)
    fft_fre = fft_fre[0:len(fft_fre)//2]
    #print(fft_fre)
    
    samples_in_frame = len(fft_inverse_filter_hamm)
    #fft_single_frame_rect = fft_single_frame_rect[0:samples_in_frame//2]
    fft_single_frame_hamm = fft_single_frame_hamm[0:samples_in_frame//2]
    #fft_inverse_filter_rect = fft_inverse_filter_rect[0:samples_in_frame//2]
    fft_inverse_filter_hamm = fft_inverse_filter_hamm[0:samples_in_frame//2]
    #fft_h1 = fft_h1[0:len(fft_h1)//2]
    #fft_h2 = fft_h2[0:len(fft_h2)//2]
    
    #peak_location_rect = librosa.util.peak_pick(fft_inverse_filter_rect,np.ceil(samples_in_frame/20),np.ceil(samples_in_frame/20),np.ceil(samples_in_frame/20),np.ceil(samples_in_frame/20),0.1,10)
    #peak_location_hamm = librosa.util.peak_pick(fft_inverse_filter_hamm,np.ceil(samples_in_frame/15),np.ceil(samples_in_frame/15),np.ceil(samples_in_frame/15),np.ceil(samples_in_frame/15),0.1,20)
    
    peak_location_hamm = scipy.signal.find_peaks(fft_inverse_filter_hamm)
    """
    peak_amplitude_rect = []
    #print('The formants in frame when rectangular window is considered are: ')
    for i in peak_location_rect:
        peak_amplitude_rect.append(fft_inverse_filter_rect[i])
    peak_amplitude_rect = np.asarray(peak_amplitude_rect)
    peak_location_rect = peak_location_rect/(samples_in_frame//2)
    """
    peak_amplitude_hamm = []
    #print('The formants in frame when hamming window is considered are: ')
    for i in peak_location_hamm[0]:
        peak_amplitude_hamm.append(fft_inverse_filter_hamm[i])
    peak_amplitude_hamm = np.asarray(peak_amplitude_hamm)
    peak_location_hamm = peak_location_hamm[0]/(samples_in_frame//2)
    peak_location_hamm = peak_location_hamm*(srl//2)
    
    for i in peak_location_hamm:
        print(i)
    
    #peak_amplitude_hamm = fft_inverse_filter_hamm[peak_location_hamm]
    
    #fft_fre = fft_fre/max(fft_fre)
    
    plt.figure()
    plt.plot(fft_fre,fft_single_frame_hamm)
    plt.plot(fft_fre,fft_inverse_filter_hamm)
    plt.scatter(peak_location_hamm, peak_amplitude_hamm)
    #plt.plot(w2/np.pi, fft_h2)
    plt.title('frame number {}'.format(int(frame_number)))
    plt.legend(['Signal Spectrum','LPC Spectrum ','peak_points'])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Log Magnitude Spectrum [dB]')
    plt.grid()
    plt.show()
    
    plt.pause(0.05)
