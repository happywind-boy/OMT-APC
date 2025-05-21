function Val2 = PiecewiseAffineMap(T, V, Val, V2)
if size(Val,2) == 1
    doRecover = 1;
    e = ones(1,3);
    Val = Val(:,e);
else
    doRecover = 0;
end
M     = triangulation(T, V);
M_Val = triangulation(T, Val);
[Loca, Bary] = pointLocation(M, V2);
NaNid = isnan(Loca);
NaNnum = sum(NaNid);
Val2 = 0*V2;
fprintf('Number of outer points: %d\n', NaNnum);
Val2(~NaNid,:) = barycentricToCartesian(M_Val, Loca(~NaNid), Bary(~NaNid,:));
Idx = knnsearch(V, V2(NaNid,:));
Val2(NaNid,:) = Val(Idx,:);
if doRecover
    Val = Val(:,1);
end
