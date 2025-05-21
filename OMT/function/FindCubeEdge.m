function Pid = FindCubeEdge(F, V, Cid)
%   8 --- 7 
% 5 --- 6 |
% | 4 --- 3
% 1 --- 2 
Pid = cell(8,8);
Pid{1,2} = DiscreteGeodesic(F, V, Cid(1), Cid(2));
Pid{2,3} = DiscreteGeodesic(F, V, Cid(2), Cid(3));
Pid{3,4} = DiscreteGeodesic(F, V, Cid(3), Cid(4));
Pid{4,1} = DiscreteGeodesic(F, V, Cid(4), Cid(1));
Pid{1,5} = DiscreteGeodesic(F, V, Cid(1), Cid(5));
Pid{2,6} = DiscreteGeodesic(F, V, Cid(2), Cid(6));
Pid{3,7} = DiscreteGeodesic(F, V, Cid(3), Cid(7));
Pid{4,8} = DiscreteGeodesic(F, V, Cid(4), Cid(8));
Pid{5,6} = DiscreteGeodesic(F, V, Cid(5), Cid(6));
Pid{6,7} = DiscreteGeodesic(F, V, Cid(6), Cid(7));
Pid{7,8} = DiscreteGeodesic(F, V, Cid(7), Cid(8));
Pid{8,5} = DiscreteGeodesic(F, V, Cid(8), Cid(5));

Pid{4,3} = flipud(Pid{3,4});
Pid{1,4} = flipud(Pid{4,1});
Pid{5,1} = flipud(Pid{1,5});
Pid{6,2} = flipud(Pid{2,6});
Pid{8,4} = flipud(Pid{4,8});
Pid{6,5} = flipud(Pid{5,6});
Pid{7,6} = flipud(Pid{6,7});
