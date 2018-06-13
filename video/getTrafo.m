clc
clear all
close all
% obj = VideoReader('SaltCity2.mp4');
% for k = 1 : 1  %fill in the appropriate number
%   this_frame = readFrame(obj);
%   thisfig = figure();
%   thisax = axes('Parent', thisfig);
%   image(this_frame, 'Parent', thisax);
%   title(thisax, sprintf('Frame #%d', k));
% end
% imwrite(this_frame,'frameSaltCity.png')

x=[0,0;
    0,100;
    100,100;
    100,0];

frame=imread('frameSaltCity.png');

figure
imshow(frame)
hold on
plot(x(:,1),x(:,2),'LineWidth',3)

X=ginput(4)
% X=[396.452380952381,91.6904761904763;
%    917.404761904762,160.261904761905;
%    785.976190476191,500.261904761905;
%    -25.4523809523809,281.214285714286]
plot(X(:,1),X(:,2),'LineWidth',3)

A=[x(1,1) x(1,2) 1 0 0 0;
   0 0 0  x(1,1) x(1,2) 1;
   x(2,1) x(2,2) 1 0 0 0;
   0 0 0  x(2,1) x(2,2) 1;
   x(3,1) x(3,2) 1 0 0 0;
   0 0 0  x(3,1) x(3,2) 1;
   x(4,1) x(4,2) 1 0 0 0;
   0 0 0  x(4,1) x(4,2) 1;]

y=[X(1,1);X(1,2);X(2,1);X(2,2);X(3,1);X(3,2);X(4,1);X(4,2)]

m=A\y

T=[m(1:3)';m(4:6)']

x=[T*[x ones(4,1)]']'
plot(x(:,1),x(:,2),'LineWidth',1)
