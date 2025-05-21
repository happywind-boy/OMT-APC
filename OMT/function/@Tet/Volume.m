function Vol = Volume(T, V)
a = V(T(:,1),:);
b = V(T(:,2),:);
c = V(T(:,3),:);
d = V(T(:,4),:);
Vol = -dot((a-d),cross2(b-d,c-d),2)./6;

function r = cross2(a,b)
r = [a(:,2).*b(:,3)-a(:,3).*b(:,2),...
     a(:,3).*b(:,1)-a(:,1).*b(:,3),...
     a(:,1).*b(:,2)-a(:,2).*b(:,1)];
