function FC = FaceCenter(F, V)
FC = ( V(F(:,1),:) + V(F(:,2),:) + V(F(:,3),:) ) / 3;