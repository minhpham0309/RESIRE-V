function angle_out = solid_angle(A,B,C)
%returns solid angle between 3 3-D vectors or N x 3 arrays
%https://en.wikipedia.org/wiki/Solid_angle#Tetrahedron
magA = sqrt(sum(A.^2,2));
magB = sqrt(sum(B.^2,2));
magC = sqrt(sum(C.^2,2));

% A = A ./ magA;
% B = B ./ magB;
% C = C ./ magC;

p = cross(B,C);
top = sum(A .* p,2);

bottom = magA.*magB.*magC + dot(A,B,2).*magC + dot(A,C,2).*magB + dot(B,C,2).*magA;
% bottom = 1 + dot(A,B,2) + dot(B,C,2) + dot(A,C,2);

angle_out = 2.*atan2(top,bottom);
% 
% p = cross(B,C);
% x = sum(A .* p,2);
% y = 1 + sum(A.*B,2) + sum(A.*C,2) + sum(B.*C,2);
% angle_out = 2 .* atan2(x,y);