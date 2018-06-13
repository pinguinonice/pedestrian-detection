clc
clear all
close all
% obj = VideoReader('TownCentre.mp4');
% for k = 1 : 1  %fill in the appropriate number
%   this_frame = readFrame(obj);
%   thisfig = figure();
%   thisax = axes('Parent', thisfig);
%   image(this_frame, 'Parent', thisax);
%   title(thisax, sprintf('Frame #%d', k));
% end
% imwrite(this_frame,'frame.png')

x=[0,0, 1;
    0,100,1;
    100,100,1;
    100,0,1]


X=[396.452380952381,91.6904761904763,1;
   917.404761904762,160.261904761905,1;
   785.976190476191,500.261904761905,1;
   -25.4523809523809,281.214285714286,1]

frame=imread('frameSaltCity.png');

figure
imshow(frame)
X=ginput(4)

hold on
plot(x(:,1),x(:,2),'LineWidth',3)
plot(X(:,1),X(:,2),'LineWidth',3)



v = homography_solve(x(:,1:2)', X(:,1:2)')
tform = fitgeotrans(x(:,1:2), X(:,1:2),'projective')

% Transform pixel coordinates using Homography,
x = [v * x']';

%x =[tform.T*x']

x=x./x(:,3)
plot(x(:,1),x(:,2),'LineWidth',1)

vinv=inv(v);
Xnew=[vinv*x']';

Xnew=Xnew./Xnew(:,3);
 
 