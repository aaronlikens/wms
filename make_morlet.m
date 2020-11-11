function wavelet = make_morlet(t, srate, hz, n_cycles, do_plot)
    % t = length of wavelet in seconds
    % srate = sample rate of wavelet. This should match the sample rate of
    % the signal. 
    % hz = is the peak frequency of the wavelet
    % n_cycles = the number of cycles for the wavelet.  This controls the
    % width of the Gaussian used to construct the wavelet
    % do_plot = a logical indicating whether plot should be generated.
    % create wavelet and take FFT
    wave_start    = -floor(t/2);
    wave_stop     = floor(t/2);
    time          = wave_start:1/srate:wave_stop;
    s             = n_cycles/(2*pi*hz);
    
    % should I normalize the fft for length?
    wavelet = exp(2*1i*pi*hz.*time) .* exp(-time.^2./(2*(s^2)));
    wavelet = wavelet./max(wavelet);
    
    if do_plot
        plot3(time, real(wavelet), imag(wavelet))
        xlabel('Time (s)');
        ylabel('Amplitude');
    end

end