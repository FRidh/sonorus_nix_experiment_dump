#import numpy as np
#import argparse
#from acoustics import Signal

#def fade(signal, duration, kind='linear'):
    #"""Fade signal"""
    #samples = int(duration * signal.fs)
    #if kind=='linear':
        #fade = np.linspace(0., 1., samples)
    #else:
        #raise ValueError("Unsupported")
    #signal[0:samples] *= fade
    ##signal[0:samples] = signal[0:samples] * fade
    #return signal

#def fade_both_sides(stimuli, duration_start, duration_end):
    ## Fade start
    #stimuli = fade(stimuli, duration_start)
    ## Fade end
    #stimuli = fade(stimuli[::-1], duration_end)[::-1]
    #return stimuli


#def main():

    #parser = argparse.ArgumentParser()
    #parser.add_argument("cut_start", type=float)
    #parser.add_argument("cut_end", type=float)
    #parser.add_argument("fade_start", type=float)
    #parser.add_argument("fade_end", type=float)
    #parser.add_argument("file_in", type=str)
    #parser.add_argument("file_out", type=str)
    #args = parser.parse_args()

    ## Load signal
    #signal = Signal.from_wav(args.file_in)

    ## Cut signal
    #signal = signal.pick(args.cut_start, args.cut_end)

    ## Fade signal
    #signal = fade_both_sides(signal, args.fade_start, args.fade_end)

    ## Save signal
    #signal.to_wav(args.file_out)

#if __name__ == '__main__':
    #main()
