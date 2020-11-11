

function out = log2space(minval, maxval,n)
    if nargin == 2
        n = 50;
    end
    out = 2.^linspace(log2(minval), log2(maxval),n);

end