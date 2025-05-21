function TC = Center(T,V)
TC = 0.25.*( V(T(:,1),:) + V(T(:,2),:) + V(T(:,3),:) + V(T(:,4),:) );
