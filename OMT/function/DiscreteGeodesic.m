function Vid = DiscreteGeodesic(F, V, Sid, Tid)
D = perform_fast_marching_mesh(V, F, Tid);
options.method = 'discrete';
[~, Vid] = compute_geodesic_mesh(D, V, F, Sid, options);
Vid = Vid.';
if Vid(end)==Tid
    Vid(end) = [];
end
if Vid(1)==Sid
    Vid(1) = [];
end