function [FoldingIdx, FoldingNum] = TetFolding(T, V)
Fid{1} = [2, 3, 4];
Fid{2} = [1, 4, 3];
Fid{3} = [1, 2, 4];
Fid{4} = [1, 3, 2];
TC = 0.25* ( V(T(:,1),:) + V(T(:,2),:) + V(T(:,3),:) + V(T(:,4),:) );
for ii = 4:-1:1
    FN = FaceNormal(T(:,Fid{ii}), V);
    FC = ( V(T(:,Fid{ii}(1)),:) + V(T(:,Fid{ii}(2)),:) + V(T(:,Fid{ii}(3)),:) ) / 3;
    FT = VecNormalize(TC-FC);
    IN{ii} = find( sum(FT.*FN, 2) >= 0);
end
FoldingIdx = unique([IN{1}; IN{2}; IN{3}; IN{4}]);
FoldingNum = length(FoldingIdx);
