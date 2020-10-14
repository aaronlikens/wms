function [wt_x, wt_y, wt_theta, wcoh, synch, mean_rel_phase, median_rel_phase,freqs2use] = win_synch(x, y, srate, win,...
    minfreq, maxfreq, mincycle, maxcycle, nfreqs,...
    loglinear, wavelet_length, do_plot, do_stats)
% 'win_synch' is a method for conducting windowed multiscale synchrony
% analysis between two time series. 
%
% 'x'               - is a real-valued time series vector (1 x n). n is 
%                     series length
% 'y'               - is a real-valued time series vector (1 x n). n is 
%                     series length
% 'srate'           - is the sampling rate of the series in Hz
%
% 'win'             - is the integer size of the sliding window in number
%                     of samples
% 'minfreq'         - is a postive real number indicating the mimimum 
%                     frequency to resolve
% 'maxfreq'         - is a positive real number indicating the maximum 
%                     frequency to resolve
% 'nfreqs'          - is an integer indicating the number of frequencies 
%                     between minfreq and maxfreq
% 'loglinear'       - 1 for log spaced peak frequencies, 2 for linearly
%                     spaced frequencies
% 'wavelet_length'  - real number indicating the length of the wavelet in
%                     seconds. Wavelet will be centered at zero. For 
%                     example, if wavelet_length = 4, the wavelet time 
%                     vector will be -2:1/srate:2;
% 'do_plot'         - is a logical indicating whether results should be 
%                     plotted
% 'do_stats'        - is a logical indicating whether the statistics should 
%                     be calculated. NOT YET IMPLEMENTED.
% References:
% Likens, A. D., Wiltshire, T. J. (2020). Windowed multiscale synchrony: 
% modeling time-varying and scale-localized interpersonal coordination
% dynamics. To appear in Social Cognitive & Affective Neuroscience.
%
% Hurtado, J.M., Rubchinsky, L.L., Sigvardt, K.A. (2004). Statistical 
% method for detection of phase-locking episodes in neural oscillations. 
% Journal of Neurophysiology, 91(4), 1883?98. 10.1152/jn.00853.2003
%
% Le Van Quyen, M., Foucher, J., Lachaux, J.-P., Rodriguez, E., Lutz, A.,
% Martinerie, J., & Varela, F. J. (2001). Comparison of Hilbert transform 
% and wavelet methods for the analysis of neuronal synchrony. Journal of 
% Neuroscience Methods, 111(2), 83?98.

% set up parameters for phase synchrony
nbins = floor(exp(0.626+0.4*log(win-1))); % Recommended by Le Van Quyen et al., 2001
edges = linspace(0, 1, nbins);

% wavelet and FFT parameters
wave_start    = -wavelet_length/2;
wave_stop     = -wave_start;
time          = wave_start:1/srate:wave_stop;
half_wavelet  = (length(time)-1)/2;
n_wavelet     = length(time);
n_data        = length(x);
n_convolution = n_wavelet+n_data-1;

% generate vectors frequencies and number of cycles
if loglinear == 1
    freqs2use     = (log2space(minfreq, maxfreq, nfreqs));
    num_cycles    = (log2space(mincycle, maxcycle, nfreqs));
    
else
    freqs2use     = (linspace(minfreq, maxfreq, nfreqs));
    num_cycles    = (linspace(mincycle, maxcycle, nfreqs));
end


% data FFTs
data_fft1 = fft(x, n_convolution);
data_fft2 = fft(y, n_convolution);

% initialize output matrices
phase_x = zeros(nfreqs, n_data);
phase_y = zeros(size(phase_x));
wt_x = zeros(size(phase_x));
wt_y = zeros(size(phase_y));
wt_theta = zeros(size(phase_y));
synch = zeros(nfreqs, n_data - win-1);
wcoh = zeros(size(synch));
mean_rel_phase= zeros(nfreqs, n_data - win-1);
median_rel_phase = zeros(nfreqs, n_data - win-1);

% perform morlet convolution
for fi=1:length(freqs2use)
    
    % create wavelet and take FFT
    s = num_cycles(fi)/(2*pi*freqs2use(fi));
    
    % normalize the fft for length
    wavelet_fft = fft(exp(2*1i*pi*freqs2use(fi).*time).* exp(-time.^2./(2*(s^2))),n_convolution);
    wavelet_fft = wavelet_fft./max(wavelet_fft);
    
   % phase angles from channel 1 via convolution
    convolution_result_fft = ifft(wavelet_fft.*data_fft1,n_convolution);
    convolution_result_fft = convolution_result_fft(half_wavelet+1:end-half_wavelet);
    sig1 = convolution_result_fft;
    
    % phase angles from channel 2 via convolution
    convolution_result_fft = ifft(wavelet_fft.*data_fft2,n_convolution);
    convolution_result_fft = convolution_result_fft(half_wavelet+1:end-half_wavelet);
    sig2 = convolution_result_fft;
    
    
    % compute phase angle and store wavelet transform for each signal
    phase_x(fi,:) = angle(sig1);
    phase_y(fi,:) = angle(sig2);
    
    % compute cross-wavelet coherence
    wt_theta(fi,:) = angle(sig1.*conj(sig2));
    wt_x(fi, :) = sig1;
    wt_y(fi, :) = sig2;

    
end

% scaled-windowed synchrony index
time_synch = (1:length(x))/srate;
time_synch = time_synch(win+1:end);

for i = 1:length(freqs2use)
    col_index = 1;
    for j = (win+1):length(x)
        phasediff = wt_theta(i,j-win:j);
        
        % compute windowed histograms and probabilities
        phasehist = histcounts(mod(phasediff,1),edges);
        phaseprob = phasehist./sum(phasehist);
        
        % compute entropy and maximum entropy
        s = -sum(phaseprob.*log(phaseprob+eps));
        smax = log(nbins);
        rho = (smax-s)/smax;
        
        % compute magnitude squared coherence
        mag_sq_coh = abs(mean(exp(1i*phasediff))).^2;
       
        % compute mean relative phase as circular mean
        mean_rel_phase(i, col_index) = circ_mean(phasediff);
        median_rel_phase(i, col_index) = median(phasediff);
        synch(i, col_index) = rho;
        wcoh(i, col_index) = mag_sq_coh;
        col_index = col_index + 1;
        
    end
    
end

if do_plot
    % plot wavelet transforms and spectral coherence
    
    figure
    subplot(221)
    graph_time = (1:n_data)/srate/60;
    plot(graph_time, x, 'r');
    hold on;
    plot(graph_time, y, 'b');
    hold off;
    legend('Series x', 'Series y');
    xlabel('Time (min)');
    ylabel('Amplitude (a.u.)');
    
    subplot(222)
    imagesc(graph_time, log2(freqs2use), abs(wt_x).^2);
    plot_freqs = log2space(minfreq, maxfreq,6);
    yticks(log2(plot_freqs));
    yticklabels(num2str(sprintf('%g\n',plot_freqs)));
    title('Wavelet Transform x')
    colorbar();
    xlabel('Time (min)');
    ylabel('Frequency (Hz)');
    
    subplot(223)
    imagesc(graph_time, log2(freqs2use), abs(wt_y).^2);
    plot_freqs = log2space(minfreq, maxfreq,6);
    yticks(log2(plot_freqs));
    yticklabels(num2str(sprintf('%g\n',plot_freqs)));
    colorbar()
    title('Wavelet Transform y')
    xlabel('Time (min)');
    ylabel('Frequency (Hz)');
    
    subplot(224)
    graph_time = time_synch/60;
    imagesc(graph_time, log2(freqs2use), synch);
    plot_freqs = log2space(minfreq, maxfreq,6);
    yticks(log2(plot_freqs));
    yticklabels(num2str(sprintf('%g\n',plot_freqs)));
    colorbar()
    xlabel('Time (min)')
    ylabel('Frequency (Hz)')
    title('Windowed Multiscale Synchrony')
    
end

if do_stats
    % TODO: add in surrogation test options.
    
end
end
