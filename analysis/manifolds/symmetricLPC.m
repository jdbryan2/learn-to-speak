function [ distance ] = symmetricLPC( x, y, p)
%dynamicLPC Dynamic time warping applied to the LPC distance
%   x,y: input signals (speech segment, 8kHz sampling rate is ideal)
%   p: poles in the filter representation


[xrows, xcols] = size(x);
[yrows, ycols] = size(y);

if xcols > 1 || ycols > 1
    error('LPC function requires column vectors.');
end

x_lpc = lpc(x, p);
Rx = xcorr(x);
Rx = toeplitz(Rx((0:p)+xrows));

y_lpc = lpc(y, p);
Ry = xcorr(y);
Ry = toeplitz(Ry((0:p)+yrows));

distance = 0.5*(log((y_lpc*Rx*y_lpc')/(x_lpc*Rx*x_lpc')) + ...
            log((x_lpc*Ry*x_lpc')/(y_lpc*Ry*y_lpc')));
