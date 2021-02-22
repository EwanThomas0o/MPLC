%%% The function creates an image with complex representation of data. The
%%% brightness corresponds to the field amplitude and the color indicates
%%% its phase.
function figure1=mimshowDP(M,upscale);
% M = a complex array of the data o display
% upscale = a positive integer - upscales the Matlab plot by nearest
% neighbour interpolation


Up = ones(upscale);

mAx = max(max(abs(M)));
M = M/mAx;
A=abs(M);
P=angle(M);
cp(:,:,1)=kron(min(A,1).*(cos(P-2*pi/3)/2+.5),Up);
cp(:,:,2)=kron(min(A,1).*(cos(P)/2+.5),Up);
cp(:,:,3)=kron(min(A,1).*(cos(P+2*pi/3)/2+.5),Up);

% Create figure

figure1 = figure('Color',[1 1 1]);
axes1 = axes('Visible','off','Parent',figure1,'YDir','reverse',...
    'TickDir','out',...
    'DataAspectRatio',[1 1 1]);

%colormap(hsv);

% Create axes

box(axes1,'on');
hold(axes1,'all');

imshow(cp,'Parent',axes1);